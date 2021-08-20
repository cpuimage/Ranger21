"""Ranger21 optimizer implementation."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from typing import Union, Callable, Dict

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import deserialize
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class Ranger21(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True 

    def __init__(
            self,
            learning_rate: Union[float, Callable, Dict] = 1e-3,
            beta_1: Union[float, Callable] = 0.9,
            beta_2: Union[float, Callable] = 0.999,
            beta_3: Union[float, Callable] = 0.9,
            num_epochs: int = 1000,
            steps_per_epoch: int = 100,
            epsilon: float = 1e-8,
            use_softplus: bool = False,
            beta_softplus: float = 50.0,
            eps_clipping: float = 1e-3,
            threshold_clipping: float = 1e-2,
            weight_decay: float = 1e-4,
            beta_lookahead: float = 0.5,
            lookahead_every_nth_iter=5.0,
            nb_warmup_iterations=None,
            nb_warmdown_iterations=None,
            centralize_gradients: bool = True,
            normalize_gradients: bool = True,
            name='Ranger21',
            **kwargs):
        super(Ranger21, self).__init__(name, **kwargs)
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.nb_iterations = float(num_epochs * steps_per_epoch)
        self.use_softplus = use_softplus
        self.centralize_gradients = centralize_gradients
        self.normalize_gradients = normalize_gradients
        if isinstance(learning_rate, Dict):
            learning_rate = deserialize(learning_rate)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('beta_3', beta_3)
        self.epsilon = epsilon or backend_config.epsilon()
        self._set_hyper('nb_iterations', self.nb_iterations)
        self._set_hyper('nb_warmup_iterations',
                        float(0.22 * self.nb_iterations) if nb_warmup_iterations is None else nb_warmup_iterations)
        self._set_hyper('nb_warmdown_iterations',
                        float(0.28 * self.nb_iterations) if nb_warmdown_iterations is None else nb_warmdown_iterations)
        self._set_hyper('beta_softplus', beta_softplus)
        self._set_hyper('eps_clipping', eps_clipping)
        self._set_hyper('threshold_clipping', threshold_clipping)
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('beta_lookahead', beta_lookahead)
        self._set_hyper('lookahead_every_nth_iter', lookahead_every_nth_iter)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'prev_m')
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'slow', initializer=var)
            self.add_slot(var, 'vhat')

    def _learning_rate_scheduler(self, max_learning_rate,
                                 iteration, nb_iterations, nb_warmup_iterations, nb_warmdown_iterations,
                                 one_minus_beta_2):
        """combines explore-exploit scheduling with a linear warmup"""
        warmup_scaling = math_ops.maximum(0.5 * iteration * one_minus_beta_2, iteration / nb_warmup_iterations)
        warmdown_scaling = (nb_iterations - iteration) / nb_warmdown_iterations
        scaling = math_ops.minimum(1., math_ops.minimum(warmup_scaling, warmdown_scaling))
        return scaling * max_learning_rate

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Ranger21, self)._prepare_local(var_device, var_dtype, apply_state)
        epsilon = ops.convert_to_tensor_v2(self.epsilon, var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_3_t = array_ops.identity(self._get_hyper('beta_3', var_dtype))
        nb_iterations = array_ops.identity(self._get_hyper('nb_iterations', var_dtype))
        nb_warmup_iterations = array_ops.identity(self._get_hyper('nb_warmup_iterations', var_dtype))
        nb_warmdown_iterations = array_ops.identity(self._get_hyper('nb_warmdown_iterations', var_dtype))
        beta_softplus = array_ops.identity(self._get_hyper('beta_softplus', var_dtype))
        eps_clipping = array_ops.identity(self._get_hyper('eps_clipping', var_dtype))
        threshold_clipping = array_ops.identity(self._get_hyper('threshold_clipping', var_dtype))
        weight_decay = array_ops.identity(self._get_hyper('weight_decay', var_dtype))
        beta_lookahead = array_ops.identity(self._get_hyper('beta_lookahead', var_dtype))
        lookahead_every_nth_iter = array_ops.identity(self._get_hyper('lookahead_every_nth_iter', var_dtype))
        max_learning_rate = apply_state[(var_device, var_dtype)]['lr_t']
        one_minus_beta_1_t = 1.0 - beta_1_t
        one_minus_beta_2_t = 1.0 - beta_2_t
        one_plus_beta_3_t = 1.0 + beta_3_t
        scheduled_learning_rate = self._learning_rate_scheduler(max_learning_rate=max_learning_rate,
                                                                iteration=local_step,
                                                                nb_iterations=nb_iterations,
                                                                nb_warmup_iterations=nb_warmup_iterations,
                                                                nb_warmdown_iterations=nb_warmdown_iterations,
                                                                one_minus_beta_2=one_minus_beta_2_t)
        pnm_noise_amplitude = math_ops.sqrt(math_ops.square(one_plus_beta_3_t) + math_ops.square(beta_3_t))
        beta1_squared = math_ops.square(beta_1_t)
        one_minus_beta1_squared_t = 1.0 - beta1_squared
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        apply_state[(var_device, var_dtype)].update(
            dict(
                local_step=local_step,
                scheduled_learning_rate=scheduled_learning_rate,
                epsilon=epsilon,
                pnm_noise_amplitude=pnm_noise_amplitude,
                beta_3_t=beta_3_t,
                beta1_squared=beta1_squared,
                one_minus_beta1_squared_t=one_minus_beta1_squared_t,
                one_plus_beta_3_t=one_plus_beta_3_t,
                beta_2_t=beta_2_t,
                one_minus_beta_1_t=one_minus_beta_1_t,
                one_minus_beta_2_t=one_minus_beta_2_t,
                one_minus_beta_1_power=1.0 - beta_1_power,
                one_minus_beta_2_power=1.0 - beta_2_power,
                beta_softplus=beta_softplus,
                eps_clipping=eps_clipping,
                threshold_clipping=threshold_clipping,
                weight_decay=weight_decay,
                beta_lookahead=beta_lookahead,
                lookahead_every_nth_iter=lookahead_every_nth_iter
            ))

    def _non_zero(self, x, epsilon=1e-8, use_softplus=False, beta_softplus=50, threshold_softplus=20):
        """insures that a value is non-zero either by applying a softplus or adding an epsilon to it"""

        def smooth_softplus(x, beta):
            """
            sofplus function but with additional control over the smoothness via the beta parameter
            threshold is there for numerical stability
            """
            return array_ops.where(x > threshold_softplus, x, math_ops.softplus(beta * x) / beta)

        return smooth_softplus(x, beta_softplus) if use_softplus else math_ops.maximum(x, epsilon)

    def _axis_aware_euclidian_norm(self, var):
        """euclidian norm with special cases to deal with various layer shapes"""

        ndim = var.shape.ndims
        if ndim <= 1:
            # fully flattens the norm
            return math_ops.sqrt(math_ops.reduce_sum(math_ops.square(var), axis=None, keepdims=False))
        else:
            # dimensions along which to compute the norm, special case for linear layers
            axis = 0 if ndim <= 3 else list(range(0, ndim - 1))  # 1 ... ndim-1
            return math_ops.sqrt(math_ops.reduce_sum(math_ops.square(var), axis=axis, keepdims=True))

    def _gradient_clipping(self, grad, var, non_zero, eps=1e-3, threshold=1e-2):
        """
        variant of gradient clipping that uses a dynamic threshold
        `eps` is there to avoid freezing zero-parameters
        `non_zero` is a function that takes an input and insures that it will not be zero or negative
        """

        norm_grad = array_ops.ones_like(var) * non_zero(self._axis_aware_euclidian_norm(grad))
        norm_var = math_ops.maximum(self._axis_aware_euclidian_norm(var), eps)
        dynamic_threshold = threshold * (norm_var / norm_grad)
        return array_ops.where(dynamic_threshold < 1., grad * dynamic_threshold, grad)

    def _gradient_normalization(self, grad, non_zero, centralize_gradients=True, normalize_gradients=True):
        """
        substract the mean from the gradient and divide it by its standard deviation
        `non_zero` is a function that takes an input and insures that it will not be zero or negative
        """
        ndim = grad.shape.ndims
        can_centralize = centralize_gradients and (ndim > 1)
        size = 1
        for i in range(ndim):
            size *= grad.shape.dims[i].value  # np.prod(grad.shape.dims)
        can_normalize = normalize_gradients and (size > 2)
        if can_centralize or can_normalize:
            # takes into account the fact that the gradient might be 1D
            keepdims = (ndim > 1)
            axis = list(range(0, ndim - 1)) if keepdims else None
            # substract the mean from the gradient
            grad_mean = math_ops.reduce_mean(grad, axis=axis, keepdims=keepdims)
            grad -= grad_mean
            if can_normalize:
                # divide the centralized gradient by its standard deviation
                grad_std = math_ops.reduce_std(grad, axis=axis, keepdims=keepdims)
                grad /= non_zero(grad_std)  # we divide *after* subtracting the mean
                # add the mean back to the gradient if we don't want to centralize it
                if not can_centralize:
                    grad += grad_mean
        return grad

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super(Ranger21, self).set_weights(weights)

    def _look_ahead(self, coefficients, train_op, var):
        """lookahead at the param level instead of group level"""
        with ops.control_dependencies([train_op]):
            slow_var = self.get_slot(var, 'slow')
            step_back = slow_var + coefficients['beta_lookahead'] * (var - slow_var)
            sync_cond = math_ops.equal(
                math_ops.floordiv(coefficients['local_step'], coefficients['lookahead_every_nth_iter']) * coefficients[
                    'lookahead_every_nth_iter'], coefficients['local_step'])
            with ops.control_dependencies([step_back]):
                slow_update = state_ops.assign(slow_var, array_ops.where(sync_cond, step_back, slow_var),
                                               use_locking=self._use_locking)
                var_update = state_ops.assign(var, array_ops.where(sync_cond, step_back, var),
                                              use_locking=self._use_locking)
            look_ahead_op = control_flow_ops.group(slow_update, var_update)
        return look_ahead_op

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        non_zero = partial(self._non_zero, epsilon=coefficients['epsilon'], use_softplus=self.use_softplus,
                           beta_softplus=coefficients[
                               'beta_softplus'])
        # prepares gradient
        grad = self._gradient_clipping(grad, var, non_zero, coefficients['eps_clipping'],
                                       coefficients['threshold_clipping'])
        grad = self._gradient_normalization(grad, non_zero, self.centralize_gradients, self.normalize_gradients)
        # first moment estimation
        # using positive-negative momentum and bias correction
        prev_m = self.get_slot(var, 'prev_m')
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * coefficients['one_minus_beta1_squared_t']
        prev_m_values = coefficients['beta1_squared'] * prev_m
        prev_m_t = state_ops.assign(prev_m, m, use_locking=self._use_locking)
        m_beta = coefficients['beta_3_t'] * m
        m_t = state_ops.assign(m, prev_m_values + m_scaled_g_values, use_locking=self._use_locking)
        m_ema = coefficients['one_plus_beta_3_t'] * m_t - m_beta
        m_ema_corr = m_ema / coefficients['one_minus_beta_1_power']
        # second moment estimation
        # using positive-negative momentum and bias correction
        v = self.get_slot(var, 'v')
        v_scaled_g_values = math_ops.square(grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'] + v_scaled_g_values, use_locking=self._use_locking)
        v_hat = self.get_slot(var, 'vhat')
        v_hat_t = math_ops.maximum(v_hat, v_t)
        with ops.control_dependencies([v_hat_t]):
            v_hat_t = state_ops.assign(v_hat, v_hat_t, use_locking=self._use_locking)
        v_ema_hat_corr = v_hat_t / coefficients['one_minus_beta_2_power']
        # update vector
        # takes positive negative momentum into account
        denom = coefficients['pnm_noise_amplitude'] * math_ops.sqrt(v_ema_hat_corr)
        update = m_ema_corr / non_zero(denom)
        # weight decay
        # combining norm-loss and stable weight decay
        euclidian_norm = self._axis_aware_euclidian_norm(var)  # for norm-loss regularization
        effective_stepsize_inv = math_ops.sqrt(math_ops.reduce_mean(v_ema_hat_corr))  # for stable weight decay
        scaled_weight_decay = coefficients['weight_decay'] * (euclidian_norm - 1.) / non_zero(
            euclidian_norm * effective_stepsize_inv)
        update += scaled_weight_decay * var
        # applies update
        var_update = state_ops.assign_sub(var, update * coefficients['scheduled_learning_rate'],
                                          use_locking=self._use_locking)
        updates = [prev_m_t, m_t, v_t, v_hat_t, var_update]
        train_op = control_flow_ops.group(*updates)
        look_ahead_op = self._look_ahead(coefficients, train_op, var)
        return control_flow_ops.group(train_op, look_ahead_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError()

    def get_config(self):
        config = super(Ranger21, self).get_config()
        config.update(
            {
                'learning_rate': self._serialize_hyperparameter('learning_rate'),
                'beta_1': self._serialize_hyperparameter('beta_1'),
                'beta_2': self._serialize_hyperparameter('beta_2'),
                'beta_3': self._serialize_hyperparameter('beta_3'),
                'nb_iterations': self._serialize_hyperparameter('nb_iterations'),
                'epsilon': self.epsilon,
                'num_epochs': self.num_epochs,
                'steps_per_epoch': self.steps_per_epoch,
                'use_softplus': self.use_softplus,
                'centralize_gradients': self.centralize_gradients,
                'normalize_gradients': self.normalize_gradients,
                'nb_warmup_iterations': self._serialize_hyperparameter('nb_warmup_iterations'),
                'nb_warmdown_iterations': self._serialize_hyperparameter('nb_warmdown_iterations'),
                'beta_softplus': self._serialize_hyperparameter('beta_softplus'),
                'eps_clipping': self._serialize_hyperparameter('eps_clipping'),
                'threshold_clipping': self._serialize_hyperparameter('threshold_clipping'),
                'weight_decay': self._serialize_hyperparameter('weight_decay'),
                'beta_lookahead': self._serialize_hyperparameter('beta_lookahead'),
                'lookahead_every_nth_iter': self._serialize_hyperparameter('lookahead_every_nth_iter'),
            }
        )
        return config
