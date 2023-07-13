use yaif::{nn::NN, data_set::DataSet};
use rand::Rng;

fn xor() {
    match NN::new(vec![2, 2, 1]) {
        Ok(mut nn) => {
            let inputs: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
            let outputs: Vec<Vec<f32>> = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
            let data = DataSet::new(&inputs, &outputs);
            nn.initialize_data(&data);
            const batch_size: usize = 50;
            const training_data: usize = 10_000;
            const epoch_size: usize = training_data / batch_size;
            const epoch_count: usize = 100;
            'epoch_loop: for _ in 0..epoch_count {
                for _ in 0..epoch_size {
                    for _ in 0..batch_size {
                        nn.train();
                    }
                    nn.adapt_weights();
                }
                if nn.is_finished() {
                    nn.print_guess();
                    println!("The Neural Network has mastered the XOR!");
                    break 'epoch_loop;
                }
            }
        },
        Err(e) => {
            panic!("{}", e);
        }
    }
}
fn main() {
    xor();
}