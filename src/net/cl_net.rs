use crate::{cl_kernel::ClStruct, data_set::DataSet};

use opencl3::{memory::Buffer, error_codes::ClError};

use super::NetTrait;

pub struct ClNet {
    // General stuff
    layer_lengths: Vec<usize>,
    layer_count: usize,
    last_layer: usize,

    // The magic, but this time with GPU buffers
    layers: Vec<Option<Buffer<f32>>>,
    weights: Vec<Option<Buffer<f32>>>,
    bias: Vec<Option<Buffer<f32>>>,
    errors: Vec<Option<Buffer<f32>>>,
    pre_activation: Vec<Option<Buffer<f32>>>,

    avg_weight: Vec<Option<Buffer<f32>>>,
    avg_bias: Vec<Option<Buffer<f32>>>,
    bias_gradient: Vec<Option<Buffer<f32>>>,
    weight_gradient: Vec<Option<Buffer<f32>>>,

    // For Validation
    target_vector: Option<Buffer<f32>>,
    current_cost: Option<Buffer<f32>>,
    floating_average: f32,

    // Training Data
    training_data: DataSet,
    validation_data: DataSet,

    // OpenCL handler
    cl_struct: ClStruct
}

impl Default for ClNet {
    fn default() -> Self {
        Self {
            layer_lengths: vec![],
            layer_count: 0,
            last_layer: 0,

            layers: vec![],
            weights: vec![],
            bias: vec![],
            errors: vec![],
            pre_activation: vec![],

            avg_weight: vec![],
            avg_bias: vec![],

            bias_gradient: vec![],
            weight_gradient: vec![],

            target_vector: None,
            current_cost: None,
            floating_average: 0.0,

            training_data: DataSet::default(),
            validation_data: DataSet::default(),

            cl_struct: ClStruct::new().unwrap()
        }
    }
}

impl NetTrait for ClNet {
    fn initialize_layers(&mut self, layer_sizes: Vec<usize>) {
        self.layer_lengths = layer_sizes.clone();
        self.layer_count = self.layer_lengths.len();
        self.last_layer = self.layer_count - 1;

        self.layers = vec![];
        for _ in 0..self.layer_count { self.layers.push(None); }
       
        self.weights = vec![];
        for _ in 0..self.layer_count { self.weights.push(None); }
        
        self.bias = vec![];
        for _ in 0..self.layer_count { self.bias.push(None); }

        self.errors = vec![];
        for _ in 0..self.layer_count { self.errors.push(None); }
        
        self.pre_activation = vec![];
        for _ in 0..self.layer_count { self.pre_activation.push(None); }
        
        self.avg_weight = vec![];
        for _ in 0..self.layer_count { self.avg_weight.push(None); }
        
        self.avg_bias = vec![];
        for _ in 0..self.layer_count { self.avg_bias.push(None); }

        self.bias_gradient = vec![];
        for _ in 0..self.layer_count { self.bias_gradient.push(None); }
        
        self.weight_gradient = vec![];
        for _ in 0..self.layer_count { self.weight_gradient.push(None); }

        self.target_vector = self.cl_struct.create_buffer(self.layer_lengths[self.last_layer], 1);
        self.current_cost = self.cl_struct.create_buffer(self.layer_lengths[self.last_layer], 1);
        self.floating_average = 0.0;
    }

    fn initialize_training_data(&mut self, data: &DataSet) {
        self.training_data = data.clone();
    }

    fn initialize_validation_data(&mut self, data: &DataSet) {
        self.validation_data = data.clone();
    }

    fn is_finished(&mut self) -> bool {
        todo!()
    }
    fn get_average_accuracy(&mut self) -> f32 {
        todo!()
    }

    fn run(&mut self, debug: bool) -> bool {
        todo!()
    }

    fn learn(&mut self) {
        todo!()
    }
    fn guess(&mut self, input: &Vec<f32>) -> usize {
        todo!()
    }
    
    fn set_target(&mut self, target: &Vec<f32>) {
        todo!()
    }
    fn set_input(&mut self, input: &Vec<f32>) {
        todo!()
    }

    fn calculate_cost(&mut self) {
        todo!()
    }
    fn calculate_layers(&mut self) {
        todo!()
    }

    fn print_guess(&mut self) {
        todo!()
    }
}

impl ClNet {
    pub fn new(cl_struct: ClStruct, layer_sizes: Vec<usize>) -> Result<Self, &'static str> {
        if layer_sizes.len() == 1 {
            Err("Network only has one layer.")
        } else if layer_sizes.len() < 1 {
            Err("Network has no layers.")
        } else {
            let mut result: ClNet = Default::default();
            result.set_cl_struct(cl_struct);
            result.load_kernels();
            result.initialize_layers(layer_sizes);
            result.initialize_network()?;
            // println!("Initialized a Neural Network!\n{:?}", result);
            Ok(result)
        }
    }
    
    fn initialize_network(&mut self) -> Result<(), ClError> {
        for i in 0..self.layer_count {
            let len = self.layer_lengths[i];

            self.layers[i] = self.cl_struct.create_buffer(len, 1);
            self.errors[i] = self.cl_struct.create_buffer(len, 1);
            self.pre_activation[i] = self.cl_struct.create_buffer(len, 1);

            self.bias_gradient[i] = self.cl_struct.create_buffer(len, 1);
        }

        for i in 1..self.layer_count {
            let prev_len = self.layer_lengths[i - 1];
            let curr_len = self.layer_lengths[i];

            self.weights[i] = self.cl_struct.create_buffer(curr_len, prev_len);

            let two_over_input_count = 2.0 / (prev_len as f32);
            self.fill_buffer_gauss(&self.weights[i], curr_len * prev_len, 0.0, two_over_input_count)?;

            self.bias[i] = self.cl_struct.create_buffer(curr_len, 1);
            self.fill_buffer(&self.bias[i], curr_len, 0.1)?;
        }

        for i in 1..self.layer_count {
            let prev_len = self.layer_lengths[i - 1];
            let curr_len = self.layer_lengths[i];
            let r1 = self.cl_struct.read_buffer(&self.weights[i], prev_len * curr_len)?;
            let r2 = self.cl_struct.read_buffer(&self.bias[i], curr_len)?;
            println!("{:?}\n{:?}", r1, r2);
        }

        for i in 1..self.layer_count {
            let prev_len = self.layer_lengths[i - 1];
            let curr_len = self.layer_lengths[i];
            let (b_row, b_col) = (curr_len, 1);
            let (w_row, w_col) = (curr_len, prev_len);

            self.avg_bias[i] = self.cl_struct.create_buffer(b_row, b_col);
            self.avg_weight[i] = self.cl_struct.create_buffer(w_row, w_col);
            self.weight_gradient[i] = self.cl_struct.create_buffer(w_row, w_col);
        }
        Ok(())
    }

    fn load_kernels(&mut self) {
        self.cl_struct.load_kernels();
    }

    fn set_cl_struct(&mut self, cl_struct: ClStruct) {
        self.cl_struct = cl_struct
    }

    fn fill_buffer(&self, buffer: &Option<Buffer<f32>>, size: usize, val: f32) -> Result<(), ClError> {
        self.cl_struct.fill(buffer, size, val)
    }

    fn fill_buffer_gauss(&self, buffer: &Option<Buffer<f32>>, size: usize, mean: f32, variance: f32) -> Result<(), ClError> {
        self.cl_struct.fill_gauss(buffer, size, mean, variance)
    }
}