import numpy as np
from faker import Faker

# Generate id, name, address, department, salary
fake = Faker()
majors = ['Computer Science', 'Marketing', 'Intl Relation', 'DS&BA', 'Buz Mgmt']
schools = ["UOW", "UOL", "UB", "UOB", "RMIT"]
NUM_ROWS = 100

with open('student.csv', 'w') as f:
    for i in range(NUM_ROWS):
        name = fake.name()
        school = np.random.choice(schools)
        major = np.random.choice(majors)

        gpa1 = np.random.randint(50, 100)
        gpa2 = np.random.randint(50, 100)
        gpa3 = np.random.randint(50, 100)
        gpa4 = np.random.randint(50, 100)
        gpa5 = np.random.randint(50, 100)

        line = f'{name},{school},{major},{gpa1},{gpa2},{gpa3},{gpa4},{gpa5}\n'

        f.write(line)

print('Done')
