import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import db_manager

async def main():
    await db_manager.initialize()
    user_id = '11111111-1111-1111-1111-111111111111'
    await db_manager.execute(
        "INSERT INTO users (id, email, profile) VALUES ($1, $2, $3) ON CONFLICT (id) DO NOTHING",
        user_id, f'test-{user_id[:8]}@example.com', '{}',
    )
    row = await db_manager.fetchrow("SELECT id, email FROM users WHERE id=$1", user_id)
    print(f'user ready: {row}')
    await db_manager.close()

asyncio.run(main())