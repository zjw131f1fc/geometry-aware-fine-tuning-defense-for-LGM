"""
Batch renderer for Objaverse objects in OmniObject3D format.
"""

import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random


class ObjaverseBatchRenderer:
    """Batch renderer for Objaverse GLB files."""

    def __init__(
        self,
        blender_path: str,
        render_script_path: str,
        glb_dir: str,
        output_dir: str,
        num_views: int = 100,
        resolution: int = 800,
        num_workers: int = 4,
        fast_mode: bool = False,
    ):
        self.blender_path = Path(blender_path)
        self.render_script_path = Path(render_script_path)
        self.glb_dir = Path(glb_dir)
        self.output_dir = Path(output_dir)
        self.num_views = num_views
        self.resolution = resolution
        self.num_workers = num_workers
        self.fast_mode = fast_mode

        # Validate paths
        if not self.blender_path.exists():
            raise FileNotFoundError(f"Blender not found: {self.blender_path}")
        if not self.render_script_path.exists():
            raise FileNotFoundError(f"Render script not found: {self.render_script_path}")
        if not self.glb_dir.exists():
            raise FileNotFoundError(f"GLB directory not found: {self.glb_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_glb_files(self, num_objects: int = None, random_sample: bool = False):
        """Get list of GLB files to render."""
        all_glbs = list(self.glb_dir.glob("*.glb"))
        print(f"Found {len(all_glbs)} GLB files in {self.glb_dir}")

        if num_objects is None:
            return all_glbs

        num_objects = min(num_objects, len(all_glbs))

        if random_sample:
            return random.sample(all_glbs, num_objects)
        else:
            return all_glbs[:num_objects]

    def render_single(self, glb_path: Path, timeout: int = 600):
        """Render a single GLB file."""
        object_uid = glb_path.stem

        # Check if already rendered
        output_path = self.output_dir / object_uid / "render" / "transforms.json"
        if output_path.exists():
            return {"status": "skipped", "uid": object_uid, "message": "Already exists"}

        try:
            cmd = [
                "xvfb-run",  # Use virtual framebuffer
                "-a",  # Automatically select a free server number
                str(self.blender_path),
                "-b",  # Background mode
                "-P", str(self.render_script_path),
                "--",
                "--object_path", str(glb_path),
                "--output_dir", str(self.output_dir),
                "--object_uid", object_uid,
                "--num_views", str(self.num_views),
                "--resolution", str(self.resolution),
            ]

            # Add fast mode flag if enabled
            if self.fast_mode:
                cmd.append("--fast")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Save full output to log file
            log_dir = self.output_dir / object_uid / "render"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "render.log"
            with open(log_file, 'w') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            if result.returncode == 0:
                # Check debug info for issues
                debug_file = log_dir / "debug_info.txt"
                debug_info = {}
                if debug_file.exists():
                    with open(debug_file, 'r') as f:
                        for line in f:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                debug_info[key.strip()] = value.strip()

                # Check for empty renders
                empty_count = 0
                if "Empty/problematic renders" in debug_info:
                    try:
                        empty_count = int(debug_info["Empty/problematic renders"])
                    except:
                        pass

                return {
                    "status": "success",
                    "uid": object_uid,
                    "empty_renders": empty_count,
                    "debug_info": debug_info
                }
            else:
                error_msg = result.stderr[-200:] if result.stderr else "Unknown error"
                return {"status": "failed", "uid": object_uid, "message": error_msg}

        except subprocess.TimeoutExpired:
            return {"status": "timeout", "uid": object_uid, "message": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"status": "error", "uid": object_uid, "message": str(e)}

    def render_batch(self, num_objects: int = None, random_sample: bool = False):
        """Render multiple objects in parallel."""
        glb_files = self.get_glb_files(num_objects, random_sample)

        print(f"\n{'='*60}")
        print(f"Batch Rendering Configuration")
        print(f"{'='*60}")
        print(f"Objects to render: {len(glb_files)}")
        print(f"Views per object: {self.num_views}")
        print(f"Resolution: {self.resolution}x{self.resolution}")
        print(f"Parallel workers: {self.num_workers}")
        print(f"Fast mode: {self.fast_mode}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        # Render in parallel
        results = {"success": 0, "failed": 0, "skipped": 0, "timeout": 0, "error": 0}
        objects_with_empty_renders = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.render_single, glb_path): glb_path
                for glb_path in glb_files
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
                result = future.result()
                status = result["status"]
                results[status] = results.get(status, 0) + 1

                # Print result
                if status == "success":
                    empty_count = result.get("empty_renders", 0)
                    if empty_count > 0:
                        print(f"⚠ {result['uid']} (success but {empty_count} empty renders)")
                        objects_with_empty_renders.append({
                            "uid": result['uid'],
                            "empty_count": empty_count,
                            "debug_info": result.get("debug_info", {})
                        })
                    else:
                        print(f"✓ {result['uid']}")
                elif status == "skipped":
                    print(f"⊘ {result['uid']} (already exists)")
                else:
                    print(f"✗ {result['uid']}: {result.get('message', 'Unknown error')}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"Rendering Summary")
        print(f"{'='*60}")
        print(f"Success: {results['success']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Failed: {results['failed']}")
        print(f"Timeout: {results['timeout']}")
        print(f"Error: {results['error']}")
        print(f"{'='*60}\n")

        # Report objects with empty renders
        if objects_with_empty_renders:
            print(f"\n{'='*60}")
            print(f"Objects with Empty Renders: {len(objects_with_empty_renders)}")
            print(f"{'='*60}")
            for obj in objects_with_empty_renders:
                print(f"\n{obj['uid']}:")
                print(f"  Empty renders: {obj['empty_count']}")
                if obj['debug_info']:
                    print(f"  Scale factor: {obj['debug_info'].get('Scale factor', 'N/A')}")
                    print(f"  Total views: {obj['debug_info'].get('Total views', 'N/A')}")
            print(f"{'='*60}\n")

            # Save problematic objects list
            problem_file = self.output_dir / "objects_with_empty_renders.txt"
            with open(problem_file, 'w') as f:
                f.write("Objects with empty/problematic renders:\n\n")
                for obj in objects_with_empty_renders:
                    f.write(f"{obj['uid']}: {obj['empty_count']} empty renders\n")
                    if obj['debug_info']:
                        for key, value in obj['debug_info'].items():
                            f.write(f"  {key}: {value}\n")
                    f.write("\n")
            print(f"Problematic objects list saved to: {problem_file}")

        return results
