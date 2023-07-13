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
            Err("Network only has one layer.")
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

    pub fn learn(&mut self) {
        // Feedforward
        for i in 1..self.layer_count {
            self.calculate_layer(i);
        }

        self.calculate_cost();

        let activated = self.derivative_activation(self.last_layer);
        self.errors[self.last_layer] = self.current_cost.hadamard_product(&activated).unwrap();

        // Backpropagate
        for i in (1..(self.layer_count - 1)).rev() {
            self.transposed_weights[i + 1] = self.weights[i + 1].transpose();
            self.errors[i] = self.transposed_weights[i + 1].multiply(&self.errors[i + 1]).unwrap();
            let activated = self.derivative_activation(i);
            self.errors[i] = self.errors[i].hadamard_product(&activated).unwrap();
        }

        // Gradiant Weights
        for i in 1..self.weights.len() {
            self.transposed_layers[i - 1] = self.layers[i - 1].transpose();
            self.weight_gradient[i] = self.errors[i].dyadic_product(&self.transposed_layers[i - 1]).unwrap();
        }

        for i in 0..self.layer_count {
            self.bias_gradient[i] = self.errors[i].clone();
        }

        for i in 0..self.layer_count {
            self.avg_bias[i] = self.avg_bias[i].add(&self.bias_gradient[i]).unwrap();
            self.avg_weight[i] = self.avg_weight[i].add(&self.weight_gradient[i]).unwrap();
        }
    }

    fn calculate_cost(&mut self) {
        unimplemented!("calculate_cost");
    }

    fn calculate_layer(&mut self, layer: usize) {
        unimplemented!("calculate_layer");
    }

    fn derivative_activation(&mut self, layer: usize) -> Matrix {
        unimplemented!("derivative activation");
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    const TEST_CASES: usize = 1_000;
    #[test]
    fn init_nn() {
        for _ in 0..TEST_CASES {
            let big_net = NN::new(vec![10, 15, 20, 25, 30]);
            assert!(big_net.is_ok());

            let big_net = big_net.unwrap();

            assert_eq!(big_net.weights[0].get_dim(), (0, 0));
            assert_eq!(big_net.weights[1].get_dim(), (15, 10));
            assert_eq!(big_net.weights[2].get_dim(), (20, 15));
            assert_eq!(big_net.weights[3].get_dim(), (25, 20));
            assert_eq!(big_net.weights[4].get_dim(), (30, 25));

            assert_eq!(big_net.bias[0].get_dim(), (0, 0));
            assert_eq!(big_net.bias[1].get_dim(), (15, 1));
            assert_eq!(big_net.bias[2].get_dim(), (20, 1));
            assert_eq!(big_net.bias[3].get_dim(), (25, 1));
            assert_eq!(big_net.bias[4].get_dim(), (30, 1));

            assert_eq!(big_net.target_vector.get_dim(), (30, 1));

            let n = NN::new(vec![10, 3]);
            assert!(n.is_ok());

            let one_layer = NN::new(vec![0]);
            assert_eq!(one_layer.err().unwrap(), "Network only has one layer.");

            let no_layer = NN::new(vec![]);
            assert_eq!(no_layer.err().unwrap(), "Network has no layers.");
        }
    }
}