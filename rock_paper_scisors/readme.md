Algorithm from https://github.com/sdenton4/roshambot

Usage:

Alexa play roshambo

Simply state your move.

Say score to check the score

Algorithm:
Save history of your moves.
Look at last 2 moves you have done.
Look at what you commonly play after playing those 2 moves before
Play according to that probability distribution

A human can beat this with careful thinking but it beats fast play.

Intent Scheme
{
  "intents": [
    {
      "intent": "YesIntent"
    },
    {
      "slots": [
        {
          "name": "move",
          "type": "LIST_OF_MOVES"
        }
      ],
      "intent": "MoveIntent"
    },
    {
      "intent": "NoIntent"
    },
    {
      "intent": "ScoreIntent"
    }
  ]
}

Sample Utterances
YesIntent yes
YesIntent yeah
YesIntent ok
YesIntent sure
MoveIntent {move}
NoIntent no
NoIntent no thanks
NoIntent nope
ScoreIntent score
