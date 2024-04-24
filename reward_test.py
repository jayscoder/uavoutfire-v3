def main():
    alive_flammables = 500

    alive_near_flammable_fires = 0
    alive_fires = 10

    time_ratio = 1

    size = 50

    a, b, c = 0.1, -20, -1

    alive_flammables_ratio = alive_flammables / (size * size)
    alive_fires_ratio = alive_fires / (size * size)
    alive_near_flammable_fires_ratio = alive_near_flammable_fires / (size * size)

    reward = (alive_flammables_ratio * a
              + alive_near_flammable_fires_ratio * b
              + alive_fires_ratio * c) * time_ratio * 3000

    print(reward)


if __name__ == '__main__':
    main()
