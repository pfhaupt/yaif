use rand_distr::{Normal, Distribution};

#[derive(PartialEq, Debug, Clone, Default)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    size: usize,
    content: Vec<f32>
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let content = vec![0.0; rows * cols];
        let size = rows * cols;
        Matrix { rows, cols, size, content }
    }

    pub fn clone(&self) -> Self {
        Matrix { rows: self.rows, cols: self.cols, size: self.size, content: self.content.clone() }
    }

    #[inline]
    pub fn get_dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn get_index(&self, x: usize, y: usize) -> usize {
        x * self.cols + y
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Result<f32, &str> {
        if x >= self.rows || y >= self.cols {
            Err("Out of bounds!")
        } else {
            Ok(self.content[self.get_index(x, y)])
        }
    }

    #[inline]
    pub fn get_unchecked(&self, x: usize, y: usize) -> f32 {
        self.content[self.get_index(x, y)]
    }

    #[inline]
    pub fn get_at_index(&self, index: usize) -> f32 {
        self.content[index]
    }

    #[inline]
    pub fn get_all(&self) -> Vec<f32> {
        self.content.clone()
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, val: f32) -> Result<(), &str> {
        if x >= self.rows || y >= self.cols {
            Err("Out of bounds!")
        } else {
            let i = self.get_index(x, y);
            self.content[i] = val;
            Ok(())
        }
    }

    #[inline]
    pub fn set_unchecked(&mut self, x: usize, y: usize, val: f32) {
        let i = self.get_index(x, y);
        self.content[i] = val;
    }

    #[inline]
    pub fn set_at_index(&mut self, index: usize, val: f32) {
        self.content[index] = val;
    }

    pub fn fill(&mut self, val: f32) {
        for x in 0..self.rows {
            for y in 0..self.cols {
                self.set_unchecked(x, y, val);
            }
        }
    }

    pub fn fill_vec(&mut self, values: &Vec<f32>) {
        if values.len() != self.len() {
            panic!("Mismatched dimensions in fill_vec! {} {:?}", values.len(), self.get_dim());
        }
        for i in 0..self.size {
            self.content[i] = values[i];
        }
    }

    pub fn fill_fit(&mut self, values: &Vec<f32>, dim: usize) {
        for x in 0..self.rows {
            for y in 0..self.cols {
                self.set_unchecked(x, y, values[x * dim + y]);
            }
        }
    }

    pub fn gaussian_fill(&mut self, mean: f32, variance: f32) {
        let normal = Normal::new(mean, variance).unwrap();
        for x in 0..self.rows {
            for y in 0..self.cols {
                self.set_unchecked(x, y, normal.sample(&mut rand::thread_rng()));
            }
        }
    }

    pub fn pad(&mut self, pad_rows: usize, pad_cols: usize) {
        let old_content = self.content.clone();
        let old_rows = self.rows;
        let old_cols = self.cols;

        self.rows += pad_rows;
        self.cols += pad_cols;
        self.size = self.rows * self.cols;
        let mut new_content = vec![0.0; self.size];
        for x in 0..old_rows {
            for y in 0..old_cols {
                new_content[x * self.cols + y] = old_content[x * old_cols + y];
            }
        }
        self.content = new_content.clone();
    }

    pub fn add(&self, other: &Matrix) -> Result<Self, &str> {
        if self.rows != other.rows || self.cols != other.cols {
            Err("Mismatch in dimensions!")
        } else {
            let mut m = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    m.set_unchecked(i, j, self.get_unchecked(i, j) + other.get_unchecked(i, j));
                }
            }
            Ok(m)
        }
    }

    pub fn sub(&self, other: &Matrix) -> Result<Self, &str> {
        if self.rows != other.rows || self.cols != other.cols {
            Err("Mismatch in dimensions!")
        } else {
            let mut m = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    m.set_unchecked(i, j, self.get_unchecked(i, j) - other.get_unchecked(i, j));
                }
            }
            Ok(m)
        }
    }

    pub fn multiply(&self, other: &Matrix) -> Result<Self, &str> {
        if self.cols != other.rows {
            return Err("mismatch in dimensions!"); // Invalid matrix dimensions for multiplication
        }
        let mut result = Matrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get_unchecked(i, k) * other.get_unchecked(k, j);
                }
                result.set_unchecked(i, j, sum);
            }
        }

        Ok(result)
    }

    pub fn multiply_scalar(&mut self, scalar: f32) {
        if scalar == 1.0 {
            return;
        }
        for x in 0..self.rows {
            for y in 0..self.cols {
                self.set_unchecked(x, y, self.get_unchecked(x, y) * scalar);
            }
        }
    }

    pub fn transpose(self: &Matrix) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        let r = self.rows;
        let c = self.cols;
        for i in 0..r {
            for j in 0..c {
                result.content[j * r + i] = self.content[i * c + j];
            }
        }
        result
    }

    pub fn multiply_in_place(first: &Matrix, second: &Matrix, target: &mut Matrix) {
        if first.cols != second.rows {
            panic!("Mismatch in dimensions in multiply_in_place!");
        }
        let m = first.rows;
        let n = first.cols;
        let p = second.cols;
        for i in 0..m {
            for j in 0..p {
                let mut s = 0.0;
                for k in 0..n {
                    s += first.get_unchecked(i, k) * second.get_unchecked(k, j);
                }
                target.set_unchecked(i, j, s);
            }
        }
    }

    pub fn hadamard_product(self: &Matrix, second: &Matrix) -> Result<Matrix, &'static str> {
        if self.rows != second.rows || self.cols != second.cols {
            Err("Mismatch in dimensions at hadamard_in_place!")
        } else {
            let mut target = Matrix::new(self.rows, self.cols);
            for x in 0..self.rows {
                for y in 0..self.cols {
                    let val = self.get_unchecked(x, y) * second.get_unchecked(x, y);
                    target.set_unchecked(x, y, val);
                }
            }
            Ok(target)
        }
    }
    
    pub fn dyadic_product(self: &Matrix, second: &Matrix) -> Result<Matrix, &'static str> {
        if self.cols != 1 || second.rows != 1 {
            Err("Mismatch in dimensions at dyadic_in_place!")
        } else {
            let mut target = Matrix::new(self.rows, second.cols);
            for i in 0..self.rows {
                for j in 0..second.cols {
                    let val = self.content[i] * second.content[j];
                    target.set_unchecked(i, j, val);
                }
            }
            Ok(target)
        }
    }
}

#[cfg(test)]
mod tests {
    const TEST_CASES: usize = 10;
    use super::*;
    use rand::Rng;

    #[test]
    fn init_matrix() {
        for _ in 0..TEST_CASES {
            let r = rand::thread_rng().gen_range(0..100);
            let c = rand::thread_rng().gen_range(0..100);

            let result = Matrix::new(r, c);
            assert_eq!(result, Matrix { rows: r, cols: c, size: r * c, content: vec![0.0; r * c] });
        }
    }

    #[test]
    fn add_wrong_dim() {
        for _ in 0..TEST_CASES {
            let rows: usize = rand::thread_rng().gen_range(5..100);
            let cols: usize = rand::thread_rng().gen_range(5..100);

            let m1 = Matrix::new(rows, cols);
            let m2 = Matrix::new(cols + 1, rows + 1);
            let r1 = m1.add(&m2);
            let r2 = m2.add(&m1);

            assert!(r1.is_err());
            assert!(r2.is_err());
        }
    }

    #[test]
    fn add_correct_dim() {
        for _ in 0..TEST_CASES {
            let rows: usize = rand::thread_rng().gen_range(5..100);
            let cols: usize = rand::thread_rng().gen_range(5..100);

            let mut m1 = Matrix::new(rows, cols);
            let mut m2 = Matrix::new(rows, cols);
            let mut control_vector = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    let v1 = rand::thread_rng().gen_range(0.0..10.0);
                    let v2 = rand::thread_rng().gen_range(0.0..10.0);
                    m1.set_unchecked(i, j, v1);
                    m2.set_unchecked(i, j, v2);
                    control_vector[m1.get_index(i, j)] = v1 + v2;
                }
            }
            let result = m1.add(&m2);

            assert_eq!(result.unwrap().content, control_vector);
        }
    }

    #[test]
    fn sub_wrong_dim() {
        for _ in 0..TEST_CASES {
            let rows: usize = rand::thread_rng().gen_range(5..100);
            let cols: usize = rand::thread_rng().gen_range(5..100);

            let m1 = Matrix::new(rows, cols);
            let m2 = Matrix::new(cols + 1, rows + 1);
            let r1 = m1.sub(&m2);
            let r2 = m2.sub(&m1);

            assert!(r1.is_err());
            assert!(r2.is_err());
        }
    }

    #[test]
    fn sub_correct_dim() {
        for _ in 0..TEST_CASES {
            let rows: usize = rand::thread_rng().gen_range(5..100);
            let cols: usize = rand::thread_rng().gen_range(5..100);

            let mut m1 = Matrix::new(rows, cols);
            let mut m2 = Matrix::new(rows, cols);
            let mut control_vector = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    let v1 = rand::thread_rng().gen_range(0.0..10.0);
                    let v2 = rand::thread_rng().gen_range(0.0..10.0);
                    m1.set_unchecked(i, j, v1);
                    m2.set_unchecked(i, j, v2);
                    control_vector[m1.get_index(i, j)] = v1 - v2;
                }
            }
            let result = m1.sub(&m2);

            assert_eq!(result.unwrap().content, control_vector);
        }
    }

    #[test]
    fn mult_wrong_dim() {
        for _ in 0..TEST_CASES {
            let rows: usize = rand::thread_rng().gen_range(5..100);
            let cols: usize = rand::thread_rng().gen_range(5..100);

            let m1 = Matrix::new(rows, cols);
            let m2 = Matrix::new(cols + 1, rows + 1);
            let r1 = m1.multiply(&m2);
            let r2 = m2.multiply(&m1);

            assert!(r1.is_err());
            assert!(r2.is_err());
        }
    }

    #[test]
    fn mult_correct_dim() {
        // TODO: Add true test, this just passes as long as the dimensions are correct, hehe
        for _ in 0..TEST_CASES {
            let rows: usize = rand::thread_rng().gen_range(5..100);
            let cols: usize = rand::thread_rng().gen_range(5..100);

            let m1 = Matrix::new(rows, cols);
            let m2 = Matrix::new(cols, rows);
            let r1 = m1.multiply(&m2);
            let r2 = m2.multiply(&m1);

            assert!(r1.is_ok());
            assert!(r2.is_ok());
        }
    }

    #[test]
    fn mult_scalar() {
        for _ in 0..TEST_CASES {
            let rows: usize = rand::thread_rng().gen_range(5..100);
            let cols: usize = rand::thread_rng().gen_range(5..100);
            let scalar = rand::thread_rng().gen_range(0.0..10.0);

            let mut m1 = Matrix::new(rows, cols);
            let mut control_vector = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    let v1 = rand::thread_rng().gen_range(0.0..10.0);
                    m1.set_unchecked(i, j, v1);
                    control_vector[m1.get_index(i, j)] = v1 * scalar;
                }
            }
            m1.multiply_scalar(scalar);

            assert_eq!(m1.content, control_vector);
        }
    }
}
