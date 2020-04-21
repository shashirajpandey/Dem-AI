from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class DemPerturbedGradientDescent(optimizer.Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer from litian96/FedProx """
    def __init__(self, learning_rate=0.001, mu=0.01, use_locking=False, name="DemPGD"):
        super(DemPerturbedGradientDescent, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta = mu        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta_t = None
        self. pre_factors = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta_t = ops.convert_to_tensor(self._beta, name="prox_beta")

    def _create_slots(self, var_list):
        # Create slots for the global solution.
        for v in var_list:
            self._zeros_slot(v, "vstar", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        # vstar = self.get_slot(var, "vstar")
        # var_update = state_ops.assign_sub(var, lr_t*(grad + beta_t*(var-vstar)))  # Gradient update here:  w^{t+1}= w^{t} - lr *(grad + beta*(w^{t} - w^{0}))

        #Retrieve a pre-calculated pairs =(sum 1/N_k, sum 1/N_k * w_k)] from all fathers
        sum_Nk, sum_w_Nk = self.pre_factors
        var_update = state_ops.assign_sub(var, lr_t*(grad + 2*beta_t*(sum_Nk*var-sum_w_Nk)))  # Gradient update here:  w^{t+1}= w^{t} - lr *(grad + 2beta*(sum_Nk*w^{t} - sum_w_Nk{k=1..K})))


        return control_flow_ops.group(*[var_update,])

    
    def _apply_sparse_shared(self, grad, var, indices, scatter_add):  #only use for LSTM

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        mu_t = math_ops.cast(self._mu_t, var.dtype.base_dtype)
        vstar = self.get_slot(var, "vstar")

        v_diff = state_ops.assign(vstar, mu_t * (var - vstar), use_locking=self._use_locking)

        with ops.control_dependencies([v_diff]):  # run v_diff operation before scatter_add
            scaled_grad = scatter_add(vstar, indices, grad)
        var_update = state_ops.assign_sub(var, lr_t * scaled_grad)

        return control_flow_ops.group(*[var_update,])

    def _apply_sparse(self, grad, var):  #only use for LSTM
        return self._apply_sparse_shared(
        grad.values, var, grad.indices,
        lambda x, i, v: state_ops.scatter_add(x, i, v))
    

    def set_params(self, cog, client):
        with client.graph.as_default():
            all_vars = tf.trainable_variables()
            for variable, value in zip(all_vars, cog):
                vstar = self.get_slot(variable, "vstar")
                vstar.load(value, client.sess)

    def set_hierr_knowledge(self,pre_factors):
        self.pre_factors = pre_factors