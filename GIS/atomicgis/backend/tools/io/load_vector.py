from core.registry import register

class LoadVector:

    name = "io.load_vector"

    @staticmethod
    def execute(state, path):

        state.layers["input"] = path

        return {
            "message": f"Loaded vector {path}"
        }

register(LoadVector)