use crate::{cl_kernel::ClStruct, data_set::DataSet};

use opencl3::memory::Buffer;

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

        self.target_vector = Some(self.cl_struct.create_buffer(self.layer_lengths[self.last_layer], 1));
        self.current_cost = Some(self.cl_struct.create_buffer(self.layer_lengths[self.last_layer], 1));
        self.floating_average = 0.0;
    }

    fn initialize_network(&mut self) {
        todo!()
    }
    fn initialize_training_data(&mut self, data: &DataSet) {
        todo!()
    }
    fn initialize_validation_data(&mut self, data: &DataSet) {
        todo!()
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
            result.initialize_layers(layer_sizes);
            result.initialize_network();
            // result.set_cl_struct(cl_struct);
            // result.load_kernels();
            // println!("Initialized a Neural Network!\n{:?}", result);
            Ok(result)
        }
    }

    fn load_kernels(&mut self) {
        todo!()
    }

    fn set_cl_struct(&mut self, cl_struct: ClStruct) {
        self.cl_struct = cl_struct
    }
}