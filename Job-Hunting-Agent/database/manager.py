"""
SQLite & Storage Persistence Engine for the Job Hunting Agent.

Maintains cached job search listings, user preferences, job evaluations,
and tailored application materials outside of core module.
"""

import json
import sqlite3
from pathlib import Path

from core.deduplicator import normalize_url
from core.state import (
    ApplicationMaterials,
    JobListing,
    JobMatchAnalysis,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = Path("./database/job_hunting_agent.db")


class JobDatabase:
    """SQLite Database Manager for persisting job listings and evaluation history."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        """Initialize database manager and ensure table schema exists."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Helper to get a SQLite connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Job listings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_listings (
                    id TEXT PRIMARY KEY,
                    canonical_url TEXT UNIQUE,
                    title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    location TEXT,
                    description TEXT,
                    source TEXT,
                    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_roles TEXT,
                    preferred_locations TEXT,
                    remote_preference TEXT,
                    min_salary INTEGER,
                    experience_level TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Job match evaluation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_match_history (
                    job_id TEXT PRIMARY KEY,
                    job_title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    url TEXT,
                    overall_score REAL,
                    skills_match_score REAL,
                    experience_match_score REAL,
                    matching_skills TEXT,
                    missing_skills TEXT,
                    match_rationale TEXT,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES job_listings(id)
                );
            """)

            # Tailored application materials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS application_materials (
                    job_id TEXT PRIMARY KEY,
                    job_title TEXT NOT NULL,
                    company TEXT NOT NULL,
                    tailored_bullets TEXT,
                    cover_letter TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(job_id) REFERENCES job_listings(id)
                );
            """)

            conn.commit()
            logger.debug("Database initialized cleanly at %s ✅", self.db_path)

    def is_url_seen(self, url: str) -> bool:
        """Check if a canonical job URL has already been processed and cached."""
        canonical = normalize_url(url)
        if canonical == "N/A":
            return False

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM job_listings WHERE canonical_url = ?", (canonical,))
            return cursor.fetchone() is not None

    def save_job_listings(self, listings: list[JobListing]) -> int:
        """Persist a list of JobListing objects to the database, ignoring duplicates.

        Returns:
            int: Number of new listings inserted.
        """
        inserted_count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for job in listings:
                canonical = normalize_url(job.url)
                try:
                    cursor.execute("""
                        INSERT INTO job_listings (id, canonical_url, title, company, location, description, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(canonical_url) DO NOTHING;
                    """, (
                        job.id,
                        canonical,
                        job.title,
                        job.company,
                        job.location,
                        job.description,
                        job.source
                    ))
                    if cursor.rowcount > 0:
                        inserted_count += 1
                except sqlite3.Error as e:
                    logger.debug("Error saving job listing %s: %s", job.id, e)
            conn.commit()

        logger.info("Saved %d new job listings to SQLite database ✅", inserted_count)
        return inserted_count

    def save_job_evaluations(self, evaluations: list[JobMatchAnalysis]) -> None:
        """Persist job match evaluation results to the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for ev in evaluations:
                cursor.execute("""
                    INSERT INTO job_match_history (
                        job_id, job_title, company, url, overall_score,
                        skills_match_score, experience_match_score, matching_skills,
                        missing_skills, match_rationale
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        overall_score = excluded.overall_score,
                        match_rationale = excluded.match_rationale;
                """, (
                    ev.job_id,
                    ev.job_title,
                    ev.company,
                    ev.url,
                    ev.overall_score,
                    ev.skills_match_score,
                    ev.experience_match_score,
                    json.dumps(ev.matching_skills),
                    json.dumps(ev.missing_skills),
                    ev.match_rationale
                ))
            conn.commit()
            logger.info("Persisted %d job evaluations to SQLite database ✅", len(evaluations))

    def save_application_materials(self, materials: ApplicationMaterials) -> None:
        """Persist tailored application materials (bullets & cover letter) to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO application_materials (
                    job_id, job_title, company, tailored_bullets, cover_letter
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    tailored_bullets = excluded.tailored_bullets,
                    cover_letter = excluded.cover_letter;
            """, (
                materials.job_id,
                materials.job_title,
                materials.company,
                json.dumps(materials.tailored_bullets),
                materials.cover_letter
            ))
            conn.commit()
            logger.info("Persisted tailored application materials for job '%s' to database ✅", materials.job_id)


# Global database instance
db = JobDatabase()
