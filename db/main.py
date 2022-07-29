import asyncio
from prisma import Prisma

async def main() -> None:
    db = Prisma()
    await db.connect()

    product = await db.products.create(
        {
            'asin': '1239DASDC1',
            'name': 'Test Product Name',
            'description': 'Test product name',
            'price': 12.7,
            'categories': ['ABC', 'DEF'],
        }
    )
    print(f'created post: {product.json(indent=2, sort_keys=True)}')

    found = await db.products.find_unique(where={'id': product.id})
    assert found is not None
    print(f'found post: {found.json(indent=2, sort_keys=True)}')

    await db.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
