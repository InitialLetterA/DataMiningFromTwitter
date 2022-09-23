#CIS 600 datamining
#team 12

#Filter realtime tweets by using Streaming API
import twitter
import json
import datetime

CONSUMER_KEY = 'KvWTtZsQt0dUwDaJ88HcrVWDT'
CONSUMER_SECRET = '7Rl7h1WB4zJyfZ6Jb60E6NTZrsII6QbRKoTcEwU4TQ6XFjaDFe'
OAUTH_TOKEN = '1440314199243034635-ZSksDJjgQScm9uTSbrsGvdioFEdEuB'
OAUTH_TOKEN_SECRET = 'LrUgAOrMOu3GsOxCYDcYKi1DkIM1FfYAwrF47yDxUJd26'

def oauth_login(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET):
    # XXX: Go to http://twitter.com/apps/new to create an app and get values
    # for these credentials that you'll need to provide in place of these
    # empty string values that are defined as placeholders.
    # See https://developer.twitter.com/en/docs/basics/authentication/overview/oauth
    # for more information on Twitter's OAuth implementation.
    auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY, CONSUMER_SECRET)

    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


def main(twitter_api, keyword,):
    twitter_stream = twitter.TwitterStream(auth=twitter_api.auth)
    q = f"{keyword}"  # lang:en until:{until}  place_country:23424977
    results = twitter_stream.statuses.filter(track=q, language='en', place_country = 23424977)
    print('+' * 10)
    for i in results:
        # print(json.dumps(i))
        try:
            yield {
                'user_id': i['user']['id'],
                'username': i['user']['name'],
                'created_at': i['created_at'],
                'full_text': i['extended_tweet']['full_text'] if i.get('extended_tweet') is not None else i['retweeted_status']['extended_tweet']['full_text']
            }
        except KeyError:
            continue


twitter_api = oauth_login(CONSUMER_KEY, CONSUMER_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
keywords = 'Unemployment rate,Economic recovery,Covid economic, Employment rate, Economic collapse,Stable economy'
max_results = 50000

result = main(twitter_api, keywords)
#for o in range(0, max_results+1):
#    one = next(result)
for one in result:
    print(one)
    with open('streamingdata_economic4.json', 'a', encoding='utf-8')as f:
        f.write(json.dumps(one))
        f.write('\n')
