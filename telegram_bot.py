"""Example Telegram bot using the CalorieEstimator.

Accepts food photos (with optional captions) and replies with a
calorie/macro breakdown.

Setup:
    1. Get a Telegram bot token from @BotFather
    2. Set environment variables:
       export TELEGRAM_BOT_TOKEN="your-bot-token"
       export ANTHROPIC_API_KEY="sk-ant-..."
       export USDA_API_KEY="your-usda-key"
    3. pip install python-telegram-bot calorie-estimator
    4. python telegram_bot.py

Usage:
    - Send a food photo to the bot → get calorie estimate
    - Add a caption for better accuracy: "Fried in butter, large portion"
    - Send /help for instructions
"""

import logging
import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from calorie_estimator import CalorieEstimator

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Initialise the estimator once at module level
estimator = CalorieEstimator(
    provider="anthropic",
    apply_bias_correction=True,
    estimate_hidden_cals=True,
    include_confidence_ranges=True,
)


# ── Handlers ─────────────────────────────────────────────────


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message on /start."""
    await update.message.reply_text(
        "📸 *Calorie Estimator Bot*\n\n"
        "Send me a photo of your food and I'll estimate the calories "
        "and macros.\n\n"
        "*Tips for better accuracy:*\n"
        "• Add a caption with cooking details: _\"fried in olive oil\"_\n"
        "• Mention hidden ingredients: _\"with cheese inside\"_\n"
        "• Note portion context: _\"kids' portion\"_ or _\"I ate half\"_\n"
        "• Include reference objects in the frame (plate, fork, hand)\n\n"
        "Send /help for more info.",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help text on /help."""
    await update.message.reply_text(
        "*How it works:*\n\n"
        "1. I identify each food item in your photo\n"
        "2. I estimate portion sizes using visible reference objects\n"
        "3. I look up accurate nutrition data from the USDA database\n"
        "4. I calculate calories and macros per item and for the meal\n\n"
        "*What improves accuracy:*\n"
        "• Caption describing cooking method (grilled, fried, baked)\n"
        "• Whether oil/butter was used\n"
        "• Sauces or dressings that might be hidden\n"
        "• A plate or utensil visible in frame for scale\n\n"
        "*Known limitations:*\n"
        "• Mixed dishes (stews, curries) are harder to estimate\n"
        "• Hidden ingredients (rice under curry) may be missed\n"
        "• Calorie-dense foods (nuts, oils) have wider error margins\n"
        "• Expect ±25-35% accuracy on typical meals",
        parse_mode="Markdown",
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process a food photo and reply with calorie estimate."""
    # Send "analyzing" indicator
    await update.message.reply_chat_action("typing")

    # Get the largest available photo
    photo = update.message.photo[-1]
    file = await photo.get_file()

    # Download photo bytes
    image_bytearray = await file.download_as_bytearray()
    image_bytes = bytes(image_bytearray)

    # Get caption as the text description
    description = update.message.caption or None

    try:
        # Run the estimation pipeline
        result = await estimator.estimate(
            image=image_bytes,
            description=description,
        )

        # Format and send the response
        summary = result.format_summary()

        # Telegram has a 4096 char limit — truncate if needed
        if len(summary) > 4000:
            summary = result.format_compact()

        await update.message.reply_text(summary, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Estimation failed: {e}", exc_info=True)
        await update.message.reply_text(
            "Sorry, I couldn't analyse that image. Please try again with:\n"
            "• A clearer photo\n"
            "• Better lighting\n"
            "• The food centered in frame",
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages without a photo."""
    await update.message.reply_text(
        "Please send a *photo* of your food and I'll estimate the calories.\n\n"
        "You can add a caption to the photo for better accuracy, "
        "e.g. _\"grilled, no oil, small portion\"_.",
        parse_mode="Markdown",
    )


# ── Main ─────────────────────────────────────────────────────


def main() -> None:
    """Run the bot."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Set TELEGRAM_BOT_TOKEN environment variable")

    app = Application.builder().token(token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Start polling
    logger.info("Bot started — listening for food photos")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
