Playing 98 Cards with Deep Reinforcement Learning
=================================================


0. Motivation
-------------
I want to start to code Deep Reinforcement Learning (DRL) models with [Tensorflow](https://www.tensorflow.org/) and I need a fun task to begin with. 

1. 98 Cards
-----------
[98 Cards](https://play.google.com/store/apps/details?id=com.vdh.ninetyeight.android&hl=en) is an Android game that needs strategy.

> Your deck contains 98 cards ranging from 2 to 99.
> Distribute all of these cards on four different piles - one after another.
> The two piles on top need to be in ascending order, the two piles at the bottom need to be in descending order.
> To make it a little easier: if the difference between a card and a pile is exactly 10 you can put that card on that pile, the order doesn't matter. Use this rule to "shrink" your piles!
> Now try to get rid of as many cards as possible!

So the goal of this game is to collect as much score as possible. When everytime a player piles a card, a score will be given. But I fail to figure how how they assign the score. Therefore, I just assgin one point to each successful placement and the goal is to pile as many cards as possible.

2. File Structures
------------------
- env.py: the game simulator.
