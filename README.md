# Heartatime

A hackatime extension for your heart! <sub>(that also happens to check if you're alive or not)</sub>

## How to run it

> [!CAUTION]
> USING THIS EXTENSION WITH [HACKATIME](https://hackatime.hackclub.com/) MAY RESULT IN A BAN!

1. Install [phyphox](https://phyphox.org) on your phone.
2. Clone the repository.
3. Make sure both devices are on the same network (or just use a hotspot).
4. Open phyphox and go to "Acceleration (without g)".
5. Tap the three dots > Allow remote acccess.
6. Update the `PHYPHOX_IP` in `main.py` with the IP address shown in phyphox.
7. Run the backend with `uv sync` and then `uv run main.py`.
8. Run the frontend with `npm install` and then `node cliet.js`.
9. Open the frontend in your browser at `http://localhost:3000`.
