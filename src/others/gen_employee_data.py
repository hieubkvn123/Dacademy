import numpy as np
from faker import Faker

# Generate id, name, address, department, salary
fake = Faker()
departments = ['HR', 'Business', 'IT', 'Operation', 'R&D', 'Marketing']
NUM_ROWS = 10000

with open('employee.csv', 'w') as f:
    for i in range(NUM_ROWS):
        id = '{:08d}'.format(i+1)
        name = fake.name()
        address = fake.address().replace(',', ';').replace('\n', ' ')
        dept = np.random.choice(departments)
        salary = np.random.randint(100,999) * 10

        line = f'{id},{name},{address},{dept},{salary}\n'

        f.write(line)

print('Done')
