from core.registry import register

class BufferTool:

    name = "vector.buffer"

    @staticmethod
    def execute(state, layer_name, distance):

        return {
            "message": f"Buffered {layer_name} by {distance}"
        }

register(BufferTool)