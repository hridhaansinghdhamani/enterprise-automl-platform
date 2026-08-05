import shutil
import os


class ArtifactExporter:

    def export(

        self,

        model_path,

        export_dir="saved_models",

    ):

        os.makedirs(
            export_dir,
            exist_ok=True,
        )

        shutil.copy(

            model_path,

            os.path.join(

                export_dir,

                os.path.basename(model_path),

            ),

        )