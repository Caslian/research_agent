& "C:\Users\52465\.conda\envs\RA\python.exe" -c @"
import asyncio, asyncpg, os
async def main():
    conn = await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        database=os.getenv('POSTGRES_DB', 'innocore_ai'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'password'),
    )
    user_id = '11111111-1111-1111-1111-111111111111'
    await conn.execute(
        \"\"\"INSERT INTO users (id, email, profile) VALUES (\$1, \$2, '{}')
           ON CONFLICT (id) DO NOTHING\"\"\",
        user_id, f'test-{user_id[:8]}@example.com',
    )
    print(f'user {user_id} 已就绪')
    await conn.close()
asyncio.run(main())
"@
