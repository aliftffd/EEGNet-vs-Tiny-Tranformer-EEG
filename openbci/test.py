import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def main():
    BoardShim.enable_dev_board_logger()

    # 1. Setup Parameters
    params = BrainFlowInputParams()
    params.ip_address = "192.168.4.1"  # <--- CHANGE THIS to your Shield's IP
    params.ip_port = 12345              # Default port is usually 12345 or can be left 0 to let BrainFlow decide

    # 2. Select the Correct Board ID
    # Use BoardIds.CYTON_WIFI_BOARD for 8-channel
    # Use BoardIds.CYTON_DAISY_WIFI_BOARD for 16-channel
    board_id = BoardIds.CYTON_WIFI_BOARD.value 
    
    board = BoardShim(board_id, params)

    try:
        # 3. Connect and Start Streaming
        print("Connecting to board...")
        board.prepare_session()
        print("Connected. Starting stream...")
        board.start_stream()
        
        # 4. Record for 10 seconds
        time.sleep(10)
        
        # 5. Get Data
        # data is a numpy array [num_channels, num_samples]
        data = board.get_board_data() 
        
        print(f"Received {data.shape[1]} samples")
        print("Data shape:", data.shape)

    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        # 6. Cleanup
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            print("Session released.")

if __name__ == "__main__":
    main()
