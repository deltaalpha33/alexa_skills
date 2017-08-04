Movie bot

Usage:
What genre is Dunkirk?

When did Dunkirk come out?

What is Dunkirk's rating?

User: Give me a movie recommendation.
Alexa: What do you like?
User: I liked Star Wars
Alexa: Anything else?
User: I liked Pulp Fiction
.
.
.
User: Give me a movie recommendation.
Alexa: I recommend...

Recommendations are calculated using the movielens dataset and singular value decomposition.
Recommendations will not work for movies released in the last 2 years
Currently no way to recommend for multiple users.
All information is from tmdb



Setup:


Intent schema:
{
  "intents": [
    {
      "intent": "ConfirmaddIntent"
    },
    {
      "slots": [
        {
          "name": "movie",
          "type": "AMAZON.Movie"
        }
      ],
      "intent": "OverviewIntent"
    },
    {
      "slots": [
        {
          "name": "movie",
          "type": "AMAZON.Movie"
        }
      ],
      "intent": "GenreIntent"
    },
    {
      "slots": [
        {
          "name": "movie",
          "type": "AMAZON.Movie"
        }
      ],
      "intent": "addIntent"
    },
    {
      "slots": [
        {
          "name": "movie",
          "type": "AMAZON.Movie"
        }
      ],
      "intent": "DateIntent"
    },
    {
      "slots": [
        {
          "name": "movie",
          "type": "AMAZON.Movie"
        }
      ],
      "intent": "RatingIntent"
    },
    {
      "slots": [
        {
          "name": "movie",
          "type": "AMAZON.Movie"
        }
      ],
      "intent": "LikeIntent"
    },
    {
      "intent": "RecommendationIntent"
    },
    {
      "intent": "DeniedaddIntent"
    }
  ]
}

Sample utterances:

ConfirmaddIntent yes
ConfirmaddIntent yeah
ConfirmaddIntent ok
ConfirmaddIntent sure
OverviewIntent Tell me about {movie}
OverviewIntent Describe {movie}
OverviewIntent Give me an overview of {movie}
OverviewIntent {movie} overview
OverviewIntent {movie} description
GenreIntent Whats the genre of {movie}
GenreIntent What kind of movie is {movie}
GenreIntent {movie} genre
GenreIntent {movie} type
GenreIntent {movie} kind
addIntent add {movie}
addIntent download {movie}
DateIntent date of {movie}
DateIntent release date of {movie}
DateIntent year did {movie}
DateIntent when did {movie} come out
DateIntent when does {movie} come out
DateIntent When is {movie} coming out
DateIntent {movie} date
RatingIntent Whats the rating of {movie}
RatingIntent How good is {movie}
RatingIntent rating of {movie}
RatingIntent {movie} rating
DeniedaddIntent no
DeniedaddIntent no thanks
DeniedaddIntent nope
RecommendationIntent recommend me a movie
RecommendationIntent recommend a movie
RecommendationIntent movie recommendations
LikeIntent I like {movie}
LikeIntent I liked {movie}
LikeIntent like {movie}
LikeIntent {movie} was good