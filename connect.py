import psycopg2
from config import load_config
import pandas as pd

# default tables to dump
TABLES_TO_DUMP = [
    "auth_user",
    "projectapp_dataset",
    "projectapp_datapoint",
    "projectapp_file",
    "projectapp_project",
]


def dump_table_to_csv(cursor, table_name: str):
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(f"{table_name}.csv", index=False)


def connect(config, tables_to_dump: list[str] = TABLES_TO_DUMP):
    """Connect to the PostgreSQL database server and dump tables as CSVs"""
    try:
        with psycopg2.connect(**config) as conn:
            print("Connected to the PostgreSQL server.")

            with conn.cursor() as cursor:
                # List tables
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
                tables = cursor.fetchall()
                table_names = [t[0] for t in tables]

                print("Tables in the database:")
                for table_name in table_names:
                    print(table_name, type(table_name))

                for table in tables_to_dump:
                    dump_table_to_csv(cursor, table)

            return True

    except psycopg2.DatabaseError as error:
        print(error)
        return False


if __name__ == '__main__':
    config = load_config()
    print(config)
    connect(config)