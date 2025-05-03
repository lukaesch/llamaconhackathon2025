CREATE TABLE items(
  id INT,
  podcast_id INT,
  title TEXT,
  slug TEXT,
  description TEXT,
  pub_date_timestamp INT,
  item_pub_date TEXT
)

CREATE TABLE persons(
  id INT,
  name TEXT,
  slug TEXT,
  description TEXT
)

CREATE TABLE podcasts(
  id INT,
  title TEXT,
  slug TEXT,
  image_url TEXT,
  url TEXT,
  language TEXT
)

CREATE TABLE transcriptions(
  id INT,
  episode_id INT,
  from_ts TEXT,
  to_ts TEXT,
  text TEXT,
  person_id INT
)
