# Bots

There are currently different situations in which a player has to make a decision. These are

1. At the start of the election, the president has to propose a chancellor
2. Every alive player has to vote on whether the proposed chancellor becomes chancellor
3. The president has to discard one of the 3 drawn card
4. The chancellor has to enact one of the 2 card given to him by the president

## Fusing bots

This repository already implements a range of different bots. A bot function for one of the 5 actions is created by combining functions for each of the 3 roles:

``` python
vote_bot = bots.run.fuse(
    bots.bots.discard_true,  # role liberal
    bots.bots.discard_false, # role facist
    bots.bots.discard_false, # role hitler
)
```

In the case above presidents with role liberal will always try to discard a facist policy, while the players with either facist or hitler as role will try to discard a liberal policy.

## Running bots

After fusing bots like above for each of the 5 actions we are ready to simulate a game:

``` python
run_func = bots.run.closure(
    10, # the number of players
    30, # the length of history being recorded/available of the bots to see
    propose_bot,
    vote_bot,
    presi_bot,
    chanc_bot,
    shoot_bot,
)

# choose your random seed
key = jax.random.PRNGKey(42)

# the bots parameters: can be anything jit-able
params = {
    "propose": 0,
    "vote": 0,
    "presi": 0,
    "chanc": 0,
    "shoot": 0
}

# the state at the end of the simulation
state = run_func(key, params) 
```

## Benchmarking bots

TODO

## Creating your own bots

Bots are always called with the following keyward arguments (see `bots/run.py`)

``` python
... = some_bot(
    # PRNG state
    key=key,
    
    # actions is in {"propose", "vote", "presi", "chanc", "shoot"}
    params=params[action], 

    # what each player is supposed to see
    state=bots.mask.mask(state)
)
```

The signature of your bot should therefore look something like this

``` python
def custom_bot(key, params, state: dict):
    ...
```

Where the states values first axis represents the games history i.e. `value[i]` stands for the value at the `i`th last turn. Check the function `mask` in `bots/mask.py` for the available keys. Though the first axis from the output `mask` is ommited, because bots are only implemented for individual players.

### Output formats

1. __chancellor proposal__ unnormalized log probablities for each player including the president themselves
2. __voting__ probablity of voting yes
3. __presi__ probability of discarding a facist policy
4. __chanc__ probability of discarding a facist policy
5. __shoot__ unnormalized log probablities for each player including the president, note, that president will never shoot themselves
