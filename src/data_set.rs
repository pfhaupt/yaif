use rand::Rng;

#[derive(Default, Clone)]
pub struct Input {
    values: Vec<f32>
}

impl Input {
    fn new(val: &Vec<f32>) -> Self {
        Input { values: val.clone() }
    }

    fn generate_inputs(input: &Vec<Vec<f32>>) -> Vec<Input> {
        let mut inputs = vec![];
        for i in 0..input.len() {
            inputs.push(Input::new(&input[i]));
        }
        inputs
    }

    #[inline]
    pub fn as_vec(&self) -> Vec<f32> {
        self.values.clone()
    }
}

#[derive(Default, Clone, Debug)]
pub struct Output {
    solution_id: usize,
    values: Vec<f32>
}

impl Output {
    fn new(val: &Vec<f32>) -> Self {
        let mut max = 0.0;
        let mut id = 0;
        for i in 0..val.len() {
            if val[i] > max {
                max = val[i];
                id = i;
            }
        }
        Output { solution_id: id, values: val.clone() }
    }

    fn generate_outputs(output: &Vec<Vec<f32>>) -> Vec<Output> {
        let mut outputs = vec![];
        for i in 0..output.len() {
            outputs.push(Output::new(&output[i]));
        }
        outputs
    }

    #[inline]
    pub fn get_solution(&self) -> f32 {
        self.values[self.solution_id]
    }

    pub fn as_vec(&self) -> Vec<f32> {
        self.values.clone()
    }
}

#[derive(Default, Clone)]
pub struct DataSet {
    size: usize,
    inputs: Vec<Input>,
    outputs: Vec<Output>
}

impl DataSet {
    pub fn new(inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) -> Self {
        if inputs.len() != outputs.len() {
            panic!("Mismatched lengths for input and output data! {} inputs vs {} outputs", inputs.len(), outputs.len());
        }
        DataSet { size: inputs.len(), inputs: Input::generate_inputs(inputs), outputs: Output::generate_outputs(outputs) }
    }

    pub fn get_random_index(&self) -> usize {
        rand::thread_rng().gen_range(0..self.size)
    }

    pub fn get_input(&self, index: usize) -> Input {
        self.inputs[index].clone()
    }

    pub fn get_output(&self, index: usize) -> Output {
        self.outputs[index].clone()
    }
}