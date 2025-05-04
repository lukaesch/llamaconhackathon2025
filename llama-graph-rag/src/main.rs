mod sql_extraction;
use sql_extraction::extract_episode_data;
use sql_extraction::save_to_file;
use sqlx::sqlite::SqlitePool;
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Get episode ID from command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <episode_id> [output_file.json]", args[0]);
        std::process::exit(1);
    }

    let episode_id: i32 = args[1].parse()?;
    let output_file = if args.len() > 2 {
        &args[2]
    } else {
        "episode_data.json"
    };

    // Create a connection pool to SQLite database
    // Note: Replace "database.db" with your actual database file
    let pool = SqlitePool::connect("sqlite:hackathon.db").await?;

    // Extract episode data
    let episode_data = extract_episode_data(&pool, episode_id).await?;

    // Save to file
    save_to_file(&episode_data, output_file)?;

    println!(
        "Successfully extracted data for episode {} to '{}'",
        episode_id, output_file
    );

    Ok(())
}
