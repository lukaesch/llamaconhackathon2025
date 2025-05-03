use serde::{Deserialize, Serialize};
use sqlx::{sqlite::SqlitePool, Row};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process;

#[derive(Debug, Serialize, Deserialize)]
struct Item {
    id: i32,
    podcast_id: i32,
    title: String,
    slug: String,
    description: String,
    pub_date_timestamp: i64,
    item_pub_date: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Person {
    id: i32,
    name: String,
    slug: String,
    description: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Podcast {
    id: i32,
    title: String,
    slug: String,
    image_url: String,
    url: String,
    language: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Transcription {
    id: i32,
    episode_id: i32,
    from_ts: String,
    to_ts: String,
    text: String,
    person_id: i32,
}

#[derive(Debug, Serialize, Deserialize)]
struct EpisodeData {
    item: Item,
    podcast: Podcast,
    persons: Vec<Person>,
    transcriptions: Vec<Transcription>,
}

async fn extract_episode_data(
    pool: &SqlitePool,
    episode_id: i32,
) -> Result<EpisodeData, Box<dyn Error>> {
    // Get item data
    let item = sqlx::query_as!(Item, "SELECT * FROM items WHERE id = ?", episode_id)
        .fetch_one(pool)
        .await?;

    // Get podcast data
    let podcast = sqlx::query_as!(
        Podcast,
        "SELECT * FROM podcasts WHERE id = ?",
        item.podcast_id
    )
    .fetch_one(pool)
    .await?;

    // Get transcriptions
    let transcriptions = sqlx::query_as!(
        Transcription,
        "SELECT * FROM transcriptions WHERE episode_id = ?",
        episode_id
    )
    .fetch_all(pool)
    .await?;

    // Get unique person IDs from transcriptions
    let mut person_ids = Vec::new();
    for transcription in &transcriptions {
        if !person_ids.contains(&transcription.person_id) {
            person_ids.push(transcription.person_id);
        }
    }

    // Get persons data
    let mut persons = Vec::new();
    for person_id in person_ids {
        let person = sqlx::query_as!(Person, "SELECT * FROM persons WHERE id = ?", person_id)
            .fetch_one(pool)
            .await?;

        persons.push(person);
    }

    Ok(EpisodeData {
        item,
        podcast,
        persons,
        transcriptions,
    })
}

fn save_to_file(data: &EpisodeData, filename: &str) -> Result<(), Box<dyn Error>> {
    let path = Path::new(filename);
    let mut file = File::create(path)?;

    // Convert to JSON with pretty formatting
    let json = serde_json::to_string_pretty(data)?;

    // Write to file
    file.write_all(json.as_bytes())?;

    Ok(())
}
