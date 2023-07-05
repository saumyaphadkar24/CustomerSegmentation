import numpy as np
import pandas as pd
import random

class ABC:
    def __init__(self, data, n_solutions, n_iterations, limit):
        self.data = data
        self.n_solutions = n_solutions
        self.n_iterations = n_iterations
        self.limit = limit
        self.best_solution = None
        self.best_fitness = np.inf

    def run(self):
        solutions = self.initialize_solutions()
        for i in range(self.n_iterations):
            solutions = self.explore_solutions(solutions)
            solutions = self.update_solutions(solutions)
        return self.best_solution

    def initialize_solutions(self):
        solutions = []
        for i in range(self.n_solutions):
            r = random.randint(0, self.limit)
            f = random.randint(0, self.limit)
            m = random.randint(0, self.limit)
            solution = {'Recency': r, 'Frequency': f, 'Monetary': m}
            solution['Fitness'] = self.evaluate_fitness(solution)
            solutions.append(solution)
            if solution['Fitness'] < self.best_fitness:
                self.best_solution = solution
                self.best_fitness = solution['Fitness']
        return solutions

    def explore_solutions(self, solutions):
        for i in range(self.n_solutions):
            k = random.randint(0, self.n_solutions - 1)
            phi = random.uniform(-1, 1)
            new_solution = {}
            for key in solutions[i].keys():
                new_solution[key] = solutions[i][key] + phi * (solutions[i][key] - solutions[k][key])
                if new_solution[key] < 0:
                    new_solution[key] = 0
                if new_solution[key] > self.limit:
                    new_solution[key] = self.limit
            new_solution['Fitness'] = self.evaluate_fitness(new_solution)
            if new_solution['Fitness'] < solutions[i]['Fitness']:
                solutions[i] = new_solution
            if new_solution['Fitness'] < self.best_fitness:
                self.best_solution = new_solution
                self.best_fitness = new_solution['Fitness']
        return solutions

    def update_solutions(self, solutions):
        sorted_solutions = sorted(solutions, key=lambda k: k['Fitness'])
        best_solution = sorted_solutions[0]
        for i in range(self.n_solutions):
            r = random.randint(0, self.n_solutions - 1)
            if i == r:
                r = (r + 1) % self.n_solutions
            phi = random.uniform(-1, 1)
            new_solution = {}
            for key in solutions[i].keys():
                new_solution[key] = solutions[i][key] + phi * (best_solution[key] - solutions[i][key]) + phi * (solutions[r][key] - solutions[i][key])
                if new_solution[key] < 0:
                    new_solution[key] = 0
                if new_solution[key] > self.limit:
                    new_solution[key] = self.limit
            new_solution['Fitness'] = self.evaluate_fitness(new_solution)
            if new_solution['Fitness'] < solutions[i]['Fitness']:
                solutions[i] = new_solution
            if new_solution['Fitness'] < self.best_fitness:
                self.best_solution = new_solution
                self.best_fitness = new_solution['Fitness']
        return solutions

    def evaluate_fitness(self, solution):
        recency = solution['Recency']
        frequency = solution['Frequency']
        monetary = solution['Monetary']
        labels = self.data
        q1_recency = np.percentile(labels['Recency'], 25)
        q2_recency = np.percentile(labels['Recency'], 50)
        q3_recency = np.percentile(labels['Recency'], 75)
        q1_frequency = np.percentile(labels['Frequency'], 25)
        q2_frequency = np.percentile(labels['Frequency'], 50)
        q3_frequency = np.percentile(labels['Frequency'], 75)
        q1_monetary = np.percentile(labels['Monetary'], 25)
        q2_monetary = np.percentile(labels['Monetary'], 50)
        q3_monetary = np.percentile(labels['Monetary'], 75)
        if recency <= q1_recency:
            recency_label = 'Low'
        elif recency <= q2_recency:
            recency_label = 'Medium'
        elif recency <= q3_recency:
            recency_label = 'High'
        else:
            recency_label = 'Very High'
        if frequency <= q1_frequency:
            frequency_label = 'Low'
        elif frequency <= q2_frequency:
            frequency_label = 'Medium'
        elif frequency <= q3_frequency:
            frequency_label = 'High'
        else:
            frequency_label = 'Very High'
        if monetary <= q1_monetary:
            monetary_label = 'Low'
        elif monetary <= q2_monetary:
            monetary_label = 'Medium'
        elif monetary <= q3_monetary:
            monetary_label = 'High'
        else:
            monetary_label = 'Very High'
        return (recency_label, frequency_label, monetary_label)

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('CustomerSegmentation/khawakee_orders_2022.csv')

    # Instantiate ABC algorithm
    abc = ABC(data=data, n_solutions=50, n_iterations=100, limit=365)

    # Run ABC algorithm
    best_solution = abc.run()

    # Print best solution
    print(f'Best solution: {best_solution}')

    # Save the results to a csv file
    result = []
    for i in range(0, len(data)):
        r, f, m = abc.evaluate_solution(data.iloc[i])
        result.append({'Recency': r, 'Frequency': f, 'Monetary': m})
    result_df = pd.DataFrame(result)
    result_df.to_csv('customer_data_transformed.csv', index=False)