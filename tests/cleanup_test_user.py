import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import db_manager

async def main():
    await db_manager.initialize()
    user_id = '11111111-1111-1111-1111-111111111111'
    n = await db_manager.execute(
        "DELETE FROM user_paper_read_state WHERE user_id=$1", user_id,
    )
    print(f'cleaned: {n}')
    await db_manager.close()

asyncio.run(main())