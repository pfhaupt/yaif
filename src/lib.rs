
#[derive(PartialEq, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    content: Vec<f32>
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let content = vec![0.0; rows * cols];
        Matrix { rows, cols, content }
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
}

#[cfg(test)]
mod tests {
    const TEST_CASES: usize = 100;
    use super::*;
    use rand::Rng;

    #[test]
    fn init_matrix() {
        for _ in 0..TEST_CASES {
            let r = rand::thread_rng().gen_range(0..100);
            let c = rand::thread_rng().gen_range(0..100);
            let result = Matrix::new(r, c);
            assert_eq!(result, Matrix { rows: r, cols: c, content: vec![0.0; r * c] });
        }
    }

    #[test]
    fn add_not_same_dim() {
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
    fn add_same_dim() {
        for _ in 0..TEST_CASES {
            let rows: usize = rand::thread_rng().gen_range(5..100);
            let cols: usize = rand::thread_rng().gen_range(5..100);
            let mut m1 = Matrix::new(rows, cols);
            let mut m2 = Matrix::new(rows, cols);
            let mut control_vector = vec![0.0; rows * cols];
            for i in 0..m1.rows {
                for j in 0..m1.cols {
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
    fn sub_not_same_dim() {
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
    fn sub_same_dim() {
        for _ in 0..TEST_CASES {
            let rows: usize = rand::thread_rng().gen_range(5..100);
            let cols: usize = rand::thread_rng().gen_range(5..100);
            let mut m1 = Matrix::new(rows, cols);
            let mut m2 = Matrix::new(rows, cols);
            let mut control_vector = vec![0.0; rows * cols];
            for i in 0..m1.rows {
                for j in 0..m1.cols {
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
}
