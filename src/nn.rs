use super::matrix::Matrix;

#[derive(Default, Debug)]
pub struct NN {
    // General stuff
    layer_lengths: Vec<usize>,
    layer_count: usize,
    last_layer: usize,

    // The magic
    layers: Vec<Matrix>,
    transposed_layers: Vec<Matrix>,
    weights: Vec<Matrix>,
    transposed_weights: Vec<Matrix>,
    bias: Vec<Matrix>,
    errors: Vec<Matrix>,
    pre_activation: Vec<Matrix>,

    // For adjusting weights and gradients
    avg_weight: Vec<Matrix>,
    avg_bias: Vec<Matrix>,
    bias_gradient: Vec<Matrix>,
    weight_gradient: Vec<Matrix>,

    // For Validation
    target_vector: Matrix,
    current_cost: Matrix,
    floating_average: f32
}

impl NN {
    pub fn new(layer_sizes: Vec<usize>) -> Result<Self, &'static str> {
        if layer_sizes.len() == 1 {
            Err("Network has only one layer.")
        } else if layer_sizes.len() < 1 {
            Err("Network has no layers.")
        } else {
            let mut result: NN = Default::default();
            result.initialize_layers(layer_sizes);
            result.initialize_network();
            Ok(result)
        }
    }

    fn initialize_layers(&mut self, layer_sizes: Vec<usize>) {
        self.layer_lengths = layer_sizes.clone();
        self.layer_count = self.layer_lengths.len();
        self.last_layer = self.layer_count - 1;

        self.layers = vec![Matrix::new(0, 0); self.layer_count];
        self.transposed_layers = vec![Matrix::new(0, 0); self.layer_count];
        self.errors = vec![Matrix::new(0, 0); self.layer_count];
        self.weights = vec![Matrix::new(0, 0); self.layer_count];
        self.transposed_weights = vec![Matrix::new(0, 0); self.layer_count];
        self.bias = vec![Matrix::new(0, 0); self.layer_count];
        self.pre_activation = vec![Matrix::new(0, 0); self.layer_count];
    
        self.avg_weight = vec![Matrix::new(0, 0); self.layer_count];
        self.avg_bias = vec![Matrix::new(0, 0); self.layer_count];
        self.bias_gradient = vec![Matrix::new(0, 0); self.layer_count];
        self.weight_gradient = vec![Matrix::new(0, 0); self.layer_count];
    
        self.target_vector = Matrix::new(self.layer_lengths[self.last_layer], 1);
        self.current_cost = Matrix::new(self.layer_lengths[self.last_layer], 1);
        self.floating_average = 0.0;
    }

    fn initialize_network(&mut self) {
        for i in 0..self.layer_count {
            let len = self.layer_lengths[i];

            self.layers[i] = Matrix::new(len, 1);
            self.transposed_layers[i] = Matrix::new(1, len);
            self.errors[i] = Matrix::new(len, 1);
            self.pre_activation[i] = Matrix::new(len, 1);
        }

        self.weights[0] = Matrix::new(0, 0);
        self.transposed_weights[0] = Matrix::new(0, 0);
        self.bias[0] = Matrix::new(0, 0);

        for i in 1..self.layer_count {
            let prev_len = self.layer_lengths[i - 1];
            let curr_len = self.layer_lengths[i];

            self.weights[i] = Matrix::new(curr_len, prev_len);
            self.transposed_weights[i] = Matrix::new(prev_len, curr_len);

            let two_over_input_count = 2.0 / (prev_len as f32);
            self.weights[i].gaussian_fill(0.0, two_over_input_count);

            self.bias[i] = Matrix::new(curr_len, 1);
            self.bias[i].fill(0.1);
        }

        for i in 0..self.layer_count {
            let (b_row, b_col) = self.bias[i].get_dim();
            let (w_row, w_col) = self.weights[i].get_dim();

            self.avg_bias[i] = Matrix::new(b_row, b_col);
            self.avg_weight[i] = Matrix::new(w_row, w_col);
            self.weight_gradient[i] = Matrix::new(w_row, w_col);
        }
    }
}


#[cfg(test)]
mod tests {
    const TEST_CASES: usize = 1_000;
    #[test]
    fn init_nn() {
        assert!(true);
    }
}