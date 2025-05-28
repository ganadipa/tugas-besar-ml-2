class Starter:
    _instance = None
    _has_started = False  

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Starter, cls).__new__(cls)
        return cls._instance

    def start(self, fn):
        if not Starter._has_started:  
            try:
                fn()
                print("Starter has been initialized.")
            except Exception as e:
                print(f"Starter: Error during fn execution: {e}")
                print(
                    "Starter initialization attempted but fn failed. "
                    "Will not re-run."
                )
            finally:
                Starter._has_started = True
        else:
            print("Starter has already been initialized.")