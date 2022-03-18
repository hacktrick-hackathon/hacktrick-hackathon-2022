from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import numpy as np
import tensorflow as tf




class RllibPPOModel(TFModelV2):
    """
    Model that will map environment states to action probabilities. Will be shared across agents
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):

        super(RllibPPOModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # params we got to pass in from the call to "run"
        custom_params = model_config["custom_options"]


        ## Parse custom network params
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        d2rl = custom_params["D2RL"]
        assert type(d2rl) == bool

        ## Model inputs
        # Your input is a tensor the size of the grid with each channel representing a diffetent item as stated in the documentation
        # For example, in the channel representing a solar cell you will have an array (h x w) with 1 if a solar cell exists in this location and 0 otherwise
        self.inputs = tf.keras.Input(shape=obs_space.shape, name="observations")
        out = self.inputs

        # Implement your model architicture here using the given parameters if needed

        # This is just a dummpy layer so that the model works out of the box
        # It uses normal tf functional API and you can do the same
        out = tf.keras.layers.Flatten()(out)

        ## Model ouptus
        # Linear last layer for action distribution logits
        layer_out = tf.keras.layers.Dense(self.num_outputs)(out)
        # Linear last layer for value function branch of model
        value_out = tf.keras.layers.Dense(1)(out)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)


    def forward(self, input_dict, state=None, seq_lens=None):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])