use crate::data_set::DataSet;

pub mod nn;
pub mod cl_net;

const BATCH_SIZE: usize = 100;
const TRAINING_DATA: usize = 2_500;
const EPOCH_SIZE: usize = TRAINING_DATA / BATCH_SIZE;
const EPOCH_COUNT: usize = 100;

pub trait NetTrait {
    fn initialize_layers(&mut self, layer_sizes: Vec<usize>);
    fn initialize_training_data(&mut self, data: &DataSet);
    fn initialize_validation_data(&mut self, data: &DataSet);

    fn is_finished(&mut self) -> bool;
    fn get_average_accuracy(&mut self) -> f32;

    fn run(&mut self, debug: bool) -> bool {
        for _ in 0..EPOCH_COUNT {
            for _ in 0..EPOCH_SIZE {
                for _ in 0..BATCH_SIZE {
                    self.train();
                }
                self.adapt_weights();
            }
            if debug {
                self.print_guess();
            }
            if self.is_finished() {
                return true;
            }
        }
        false
    }

    fn learn(&mut self);
    fn guess(&mut self, input: &Vec<f32>) -> usize;
    fn train(&mut self);
    fn adapt_weights(&mut self);

    fn set_target(&mut self, target: &Vec<f32>);
    fn set_input(&mut self, input: &Vec<f32>);

    fn calculate_cost(&mut self);
    fn calculate_layers(&mut self);

    fn print_guess(&mut self);
}