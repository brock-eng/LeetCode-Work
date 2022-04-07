class NumMatrix:

    def __init__(self, matrix: list[list[int]]):
        numRows, numCols = len(matrix), len(matrix[0])
        self.pc = [[0 for _ in range(numCols)] for _ in range(numRows)]

        for row in range(numRows):
            for col in range(numCols) :
                self.pc[row][col] = matrix[row][col] + self.getValue(row - 1, col) + self.getValue(row, col - 1) - self.getValue(row - 1, col - 1)

    def getValue(self, row, col):
        if row < 0 or col < 0:
            return 0
        else:
            return self.pc[row][col]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:                                         
        return self.getValue(row2, col2) - self.getValue(row2, col1 - 1) - self.getValue(row1 - 1, col2) + self.getValue(row1 - 1, col1 - 1)


def main():
    matrix = [[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]
    matrix2 = [[-4, -5]]
    nm = NumMatrix(matrix2)
    print(nm.sumRegion(0, 0, 0, 1))

if __name__ == "__main__":  main()
