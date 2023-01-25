# Secret Hitler
----------
This is a blazingly fast implementation of the popular parlor game [`Secret Hitler`](https://www.secrethitler.com/assets/Secret_Hitler_Rules.pdf). It contains a build-in AI to play against and learn from.


### Motivation
---------

As avid players, we are always looking for new, promising strategies to win the game and analyse our oppnent's approaches, which caused some arguments about the best tactics. We couldn't find common ground for a while, so we decided - since we are programmers - to write some bots, let them play the game to find answers and/or new, (hopefully) better strategies.


### Features
-----
- BYOB: Build your own bot
	+ our comprehensive API makes it easy to design your own bot from scratch and train it
- blazingly fast
- play with or against a variety of AI-competitors
- follow the bot game
    + see what's going on thanks to our narrated history

### Installation
----
##### System requirements and download

- [ ] Make sure python 3.10 or newer is installed on your system.
    + if you're not sure about your currently installed version, run `python --version`
- [ ] Download the [github-repo](https://github.com/unitdeterminant/typed-lambda).
- [ ] Run ```pip -r install requirements.txt```   

Please note: These steps are tested under MX-21.3 and DEBIAN 11.6. Depending on your individual settings, some details might differ.


##### First steps



### API reference
----
If you want to build a bot, take a look at `README.md` at `game/bots`

### Tests
---------
This repo includes a `Test.py` file, which validates the game states for and a test file which checks the reasonability of the given bot-parameters given via standard input.
