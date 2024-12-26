import requests
import aiohttp
import asyncio
import urllib3

# Suppress only the single InsecureRequestWarning from urllib3 needed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_user_data(user_id):
    url = f"https://zara-boost-hackathon.nuwe.io/users/{user_id}"
    response = requests.get(url, verify=False)
    return response.json()

def get_user_list():
    url = "https://zara-boost-hackathon.nuwe.io/users"
    response = requests.get(url, verify=False)
    return response.json()

async def fetch_user_data(session, user_id):
    url = f"https://zara-boost-hackathon.nuwe.io/users/{user_id}"
    try:
        async with session.get(url, ssl=False, timeout=aiohttp.ClientTimeout(total=60)) as response:
            return await response.json()
    except asyncio.TimeoutError:
        print(f"Request for user {user_id} timed out.")
        return None

async def fetch_all_users_data(users_list, max_concurrent_requests=1000):
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        async def sem_fetch_user_data(user_id):
            async with semaphore:
                return await fetch_user_data(session, user_id)
        
        tasks = [sem_fetch_user_data(user_id) for user_id in users_list]
        return await asyncio.gather(*tasks)
