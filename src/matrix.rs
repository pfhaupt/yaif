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
    use super::*;
    use rand::Rng;

    fn get_test_data(which: &str) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Some random data generated using numpy
        // Represent 3x7 Matrices (7x7 for Multiplication)
        let input1 = vec![9, 6, 4, 4, 1, 0, 0, 1, 6, 0, 2, 9, 3, 0, 3, 0, 9, 1, 0, 0, 0].iter().map(|v| *v as f32).collect::<Vec<f32>>();
        let input2 = vec![3, 4, 1, 8, 8, 6, 8, 9, 7, 9, 4, 1, 4, 2, 5, 1, 8, 7, 0, 6, 9].iter().map(|v| *v as f32).collect::<Vec<f32>>();
        let output = match which {
            "add" => { vec![12, 10, 5, 12, 9, 6, 8, 10, 13, 9, 6, 10, 7, 2, 8, 1, 17, 8, 0, 6, 9] },
            "sub" => { vec![6, 2, 3, -4, -7, -6, -8, -8, -1, -9, -2, 8, -1, -2, -2, -1, 1, -6, 0, -6, -9] },
            "mul" => { vec![101, 82, 95, 124, 78, 102, 120, 21, 23, 13, 36, 33, 28, 34, 39, 13, 57, 46, 1, 40, 56, 63, 23, 90, 71, 2, 62, 85, 24, 15, 27, 45, 24, 36, 51, 86, 64, 89, 43, 9, 42, 27, 0, 0, 0, 0, 0, 0, 0]}
            "scalar" => { input1.iter().map(|v| *v as i32 * 5).collect() },
            e => unimplemented!("{}", e)
        };
        (input1, input2, output.iter().map(|v| *v as f32).collect::<Vec<f32>>())
    }

    #[test]
    fn init_matrix() {
        let r = rand::thread_rng().gen_range(0..100);
        let c = rand::thread_rng().gen_range(0..100);

        let result = Matrix::new(r, c);
        assert_eq!(result, Matrix { rows: r, cols: c, size: r * c, content: vec![0.0; r * c] });
    }

    #[test]
    fn add_wrong_dim() {
        let rows: usize = rand::thread_rng().gen_range(5..100);
        let cols: usize = rand::thread_rng().gen_range(5..100);

        let m1 = Matrix::new(rows, cols);
        let m2 = Matrix::new(cols + 1, rows + 1);
        let r1 = m1.add(&m2);
        let r2 = m2.add(&m1);

        assert!(r1.is_err());
        assert!(r2.is_err());
    }

    #[test]
    fn add_correct_dim() {
        let (i1, i2, o) = get_test_data("add");
        let mut m1 = Matrix::new(3, 7);
        m1.fill_vec(&i1);
        let mut m2 = Matrix::new(3, 7);
        m2.fill_vec(&i2);
        let m3 = m1.add(&m2).unwrap();
        let m4 = m2.add(&m1).unwrap();
        assert_eq!(m3.get_all(), o, "Matrix.add() is not working properly!");
        assert_eq!(m4.get_all(), o, "Matrix.add() is not working properly!");
        assert_eq!(m3, m4, "Matrix.add() is not assoziative!");
    }

    #[test]
    fn sub_wrong_dim() {
        let rows: usize = rand::thread_rng().gen_range(5..100);
        let cols: usize = rand::thread_rng().gen_range(5..100);

        let m1 = Matrix::new(rows, cols);
        let m2 = Matrix::new(cols + 1, rows + 1);
        let r1 = m1.sub(&m2);
        let r2 = m2.sub(&m1);

        assert!(r1.is_err());
        assert!(r2.is_err());
    }

    #[test]
    fn sub_correct_dim() {
        let (i1, i2, o) = get_test_data("sub");
        let mut m1 = Matrix::new(3, 7);
        m1.fill_vec(&i1);
        let mut m2 = Matrix::new(3, 7);
        m2.fill_vec(&i2);
        let m3 = m1.sub(&m2).unwrap();
        assert_eq!(m3.get_all(), o, "Matrix.sub() is not working properly!");
    }

    #[test]
    fn mult_wrong_dim() {
        let (i1, i2, _) = get_test_data("add");
        
        let mut m1 = Matrix::new(3, 7);
        m1.fill_vec(&i1);
        let mut m2 = Matrix::new(3, 7);
        m2.fill_vec(&i2);

        let r = m1.multiply(&m2);
        assert!(r.is_err(), "Matrix.mult() is supposed to fail for wrong dimensions!");
    }

    #[test]
    fn mult_correct_dim() {
        let (i1, i2, o) = get_test_data("mul");
        let mut m1 = Matrix::new(7, 3);
        m1.fill_vec(&i1);
        let mut m2 = Matrix::new(3, 7);
        m2.fill_vec(&i2);
        let m3 = m1.multiply(&m2).unwrap();
        assert_eq!(m3.get_all(), o, "Matrix.mult() is not working properly!");
    }

    #[test]
    fn mult_scalar() {
        let (i1, _, o) = get_test_data("scalar");
        let mut m1 = Matrix::new(7, 3);
        m1.fill_vec(&i1);
        m1.multiply_scalar(5.0);
        assert_eq!(m1.get_all(), o, "Matrix.mult_scalar() is not working properly!");
    }
}
