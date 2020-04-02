
import tensorflow as tf

#====== optimizer ======#
rms_prop_optimizer = "rms_prop_optimizer"

#====== learning_rate =======#
exponential_decay_learning_rate = "exponential_decay_learning_rate"


def build_optimizer(optimizer_config):
    optimizer_type = optimizer_config['type']
    optimizer = None
    if optimizer_type == rms_prop_optimizer:
        learning_rate_config = optimizer_config["learning_rate"]
        decay = optimizer_config['decay']
        momentum_optimizer_value = optimizer_config['momentum_optimizer_value']
        epsilon = optimizer_config['epsilon']
        learning_rate = _create_learning_rate(learning_rate_config)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                                rho = decay,
                                                momentum = momentum_optimizer_value,
                                                epsilon = epsilon)

    if optimizer is None:
        raise ValueError("optimizer not support")

    return optimizer



def _create_learning_rate(learning_rate_config):
    learning_rate_type = learning_rate_config["type"]
    learning_rate = None
    if learning_rate_type == exponential_decay_learning_rate:
        initial_learning_rate = learning_rate_config['initial_learning_rate']
        decay_steps = learning_rate_config['decay_steps']
        decay_factor = learning_rate_config['decay_factor']
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                                       decay_steps,
                                                                       decay_factor,
                                                                       staircase=True)
    if learning_rate is None:
        raise ValueError("learning rate {} not support".format(learning_rate_config))

    return learning_rate


