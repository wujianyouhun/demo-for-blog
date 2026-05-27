class ReActPlanner:

    def build(self, task):

        return {
            "steps": [
                {
                    "tool": "io.load_vector",
                    "params": {
                        "path": "data/roads.geojson"
                    }
                },
                {
                    "tool": "vector.buffer",
                    "params": {
                        "layer_name": "input",
                        "distance": 100
                    }
                }
            ]
        }