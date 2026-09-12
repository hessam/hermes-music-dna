"""aiogram FSM States for Music DNA interactive dialogs."""
from aiogram.fsm.state import State, StatesGroup


class JobFlow(StatesGroup):
    waiting_for_audio = State()    # Idle state awaiting user audio upload
    confirming_options = State()   # User selecting 30s Crisp vs 60s Full
    processing = State()           # Pipeline running in background, updates streaming
