import jax.random as jrn

from game.run import dummy_history


def vote_data(history, player: int):
    winner = 0

    #actions = history["voted"]
    pass


def main():
    key = jrn.PRNGKey(0)

    history = dummy_history(key, 5, 30)

    print(history["voted"].shape)

    vote_data(history, 0)


if __name__ == "__main__":
    # from game.run import main
    main()
