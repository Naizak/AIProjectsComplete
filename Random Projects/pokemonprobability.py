import matplotlib.pyplot as plt


def some_happening(attempts):
    y = []
    for j in attempts:
        y.append(1 - (1 - (5 / 21)) ** j)
    return y


number_of_attempts = [i for i in range(201)]
likelihood_of_happening = some_happening(number_of_attempts)

ui = int(input("How many attempts would you like to do?\n"))
for i, v in enumerate(likelihood_of_happening):
    if i == ui:
        print("The likelihood that the event will happen at least once is close to {num} percent."
              .format(num=100*(round(v, 2))))

plt.plot(number_of_attempts, likelihood_of_happening)
plt.title('Getting the sweets from the Cafe')
plt.xlabel('Number of Attempts')
plt.ylabel('Likelihood of Happening')
plt.show()
