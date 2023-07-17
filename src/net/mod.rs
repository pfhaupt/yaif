use crate::data_set::DataSet;

pub mod nn;
pub mod cl_net;

pub trait NetTrait {
    fn initialize_layers(&mut self, layer_sizes: Vec<usize>);
    fn initialize_network(&mut self);
    fn initialize_training_data(&mut self, data: &DataSet);
    fn initialize_validation_data(&mut self, data: &DataSet);

    fn is_finished(&mut self) -> bool;
    fn get_average_accuracy(&mut self) -> f32;

    fn run(&mut self, debug: bool) -> bool;

    fn learn(&mut self);
    fn guess(&mut self, input: &Vec<f32>) -> usize;

    fn set_target(&mut self, target: &Vec<f32>);
    fn set_input(&mut self, input: &Vec<f32>);

    fn calculate_cost(&mut self);
    fn calculate_layers(&mut self);

    fn print_guess(&mut self);
}