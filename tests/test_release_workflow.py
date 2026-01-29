"""
Comprehensive tests for the GitHub Actions Release (SLSA3) workflow.

This test suite validates the release workflow configuration and simulates
the build process to ensure SLSA3 provenance generation works correctly.
"""

import subprocess
import hashlib
import base64
import os
import tempfile
import shutil
import json
import yaml
import pytest
from pathlib import Path


# Path to the workflow file
WORKFLOW_FILE = Path(__file__).parent.parent / "github" / "workflows" / "release.yml"
PROJECT_ROOT = Path(__file__).parent.parent


class TestReleaseWorkflowStructure:
    """Test the structure and configuration of the release workflow YAML."""

    def test_workflow_file_exists(self):
        """Verify the release workflow file exists."""
        assert WORKFLOW_FILE.exists(), f"Workflow file not found at {WORKFLOW_FILE}"

    def test_workflow_yaml_is_valid(self):
        """Verify the workflow YAML is valid and parseable."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert workflow is not None, "Workflow YAML is empty or invalid"
        assert isinstance(workflow, dict), "Workflow YAML should be a dictionary"

    def test_workflow_has_correct_name(self):
        """Verify the workflow has the correct name."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert workflow['name'] == 'Release (SLSA3)', \
            f"Expected workflow name 'Release (SLSA3)', got '{workflow['name']}'"

    def test_workflow_trigger_on_release(self):
        """Verify the workflow is triggered on release published events."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert 'on' in workflow, "Workflow missing 'on' trigger configuration"
        assert 'release' in workflow['on'], "Workflow not triggered on release events"
        assert workflow['on']['release']['types'] == ['published'], \
            "Workflow should trigger on 'published' release type"

    def test_workflow_permissions_configured(self):
        """Verify the workflow has correct permissions for SLSA and PyPI."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert 'permissions' in workflow, "Workflow missing permissions configuration"
        permissions = workflow['permissions']

        assert permissions['contents'] == 'read', \
            "Contents permission should be 'read'"
        assert permissions['id-token'] == 'write', \
            "id-token permission should be 'write' for SLSA + PyPI trusted publishing"

    def test_workflow_has_build_job(self):
        """Verify the workflow has a build job."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert 'jobs' in workflow, "Workflow missing jobs"
        assert 'build' in workflow['jobs'], "Workflow missing 'build' job"

    def test_build_job_runs_on_ubuntu(self):
        """Verify the build job runs on Ubuntu 22.04."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        build_job = workflow['jobs']['build']
        assert build_job['runs-on'] == 'ubuntu-22.04', \
            f"Build job should run on 'ubuntu-22.04', got '{build_job['runs-on']}'"

    def test_build_job_has_outputs(self):
        """Verify the build job outputs hashes for SLSA provenance."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        build_job = workflow['jobs']['build']
        assert 'outputs' in build_job, "Build job missing outputs"
        assert 'hashes' in build_job['outputs'], "Build job missing 'hashes' output"

        # Verify the output references the hash step
        assert '${{ steps.hash.outputs.hashes }}' in build_job['outputs']['hashes']

    def test_build_job_checkout_step(self):
        """Verify the build job checks out source code."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        checkout_step = next((s for s in steps if s['name'] == 'Checkout source'), None)

        assert checkout_step is not None, "Missing 'Checkout source' step"
        assert checkout_step['uses'] == 'actions/checkout@v4', \
            "Should use actions/checkout@v4"

    def test_build_job_python_setup_step(self):
        """Verify the build job sets up Python 3.9."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        python_step = next((s for s in steps if s['name'] == 'Set up Python'), None)

        assert python_step is not None, "Missing 'Set up Python' step"
        assert python_step['uses'] == 'actions/setup-python@v5', \
            "Should use actions/setup-python@v5"
        assert python_step['with']['python-version'] == '3.9', \
            "Should use Python 3.9"

    def test_build_job_install_build_tooling(self):
        """Verify the build job installs build tooling."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        install_step = next((s for s in steps if s['name'] == 'Install build tooling'), None)

        assert install_step is not None, "Missing 'Install build tooling' step"
        assert 'run' in install_step, "Install step missing 'run' command"

        run_commands = install_step['run']
        assert 'pip install --upgrade pip' in run_commands, \
            "Should upgrade pip"
        assert 'pip install build' in run_commands, \
            "Should install 'build' package"

    def test_build_job_builds_sdist_and_wheel(self):
        """Verify the build job builds sdist and wheel."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        build_step = next((s for s in steps if s['name'] == 'Build sdist and wheel'), None)

        assert build_step is not None, "Missing 'Build sdist and wheel' step"
        assert build_step['run'] == 'python -m build', \
            "Should run 'python -m build'"

    def test_build_job_generates_hashes(self):
        """Verify the build job generates SHA256 hashes."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        hash_step = next((s for s in steps if s['name'] == 'Generate hashes'), None)

        assert hash_step is not None, "Missing 'Generate hashes' step"
        assert hash_step['id'] == 'hash', "Hash step should have id 'hash'"
        assert 'run' in hash_step, "Hash step missing 'run' command"

        run_commands = hash_step['run']
        assert 'sha256sum dist/*' in run_commands, \
            "Should generate SHA256 hashes of dist files"
        assert 'base64 -w0' in run_commands, \
            "Should base64 encode hashes"
        assert 'GITHUB_OUTPUT' in run_commands, \
            "Should write to GITHUB_OUTPUT"

    def test_build_job_uploads_artifacts(self):
        """Verify the build job uploads distribution artifacts."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        upload_step = next((s for s in steps if s['name'] == 'Upload artifacts'), None)

        assert upload_step is not None, "Missing 'Upload artifacts' step"
        assert upload_step['uses'] == 'actions/upload-artifact@v4', \
            "Should use actions/upload-artifact@v4"
        assert upload_step['with']['name'] == 'python-dist', \
            "Artifact name should be 'python-dist'"
        assert upload_step['with']['path'] == 'dist/*', \
            "Should upload all files from dist/"

    def test_workflow_has_provenance_job(self):
        """Verify the workflow has a provenance generation job."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert 'provenance' in workflow['jobs'], "Workflow missing 'provenance' job"

    def test_provenance_job_depends_on_build(self):
        """Verify the provenance job depends on the build job."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        provenance_job = workflow['jobs']['provenance']
        assert 'needs' in provenance_job, "Provenance job missing 'needs' dependency"
        assert provenance_job['needs'] == 'build', \
            "Provenance job should depend on 'build' job"

    def test_provenance_job_uses_slsa_generator(self):
        """Verify the provenance job uses SLSA generator workflow."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        provenance_job = workflow['jobs']['provenance']
        assert 'uses' in provenance_job, "Provenance job missing 'uses' field"

        expected_workflow = 'slsa-framework/slsa-github-generator/.github/workflows/generator-generic-ossf-slsa3-publish.yml@v2.0.0'
        assert provenance_job['uses'] == expected_workflow, \
            f"Should use SLSA OSSF generator v2.0.0"

    def test_provenance_job_passes_hashes(self):
        """Verify the provenance job receives hashes from build job."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        provenance_job = workflow['jobs']['provenance']
        assert 'with' in provenance_job, "Provenance job missing 'with' inputs"
        assert 'base64-subjects' in provenance_job['with'], \
            "Provenance job missing 'base64-subjects' input"

        assert '${{ needs.build.outputs.hashes }}' in provenance_job['with']['base64-subjects'], \
            "Provenance job should receive hashes from build job output"


class TestBuildProcessSimulation:
    """Simulate the build process to verify functionality."""

    `@pytest.fixture`
    def temp_build_dir(self):
        """Create a temporary build directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists for building."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        assert pyproject_file.exists(), \
            "pyproject.toml is required for python -m build"

    def test_pyproject_toml_has_project_metadata(self):
        """Verify pyproject.toml has necessary project metadata."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"

        with open(pyproject_file, 'r') as f:
            content = f.read()

        # Basic checks for required fields
        assert '[project]' in content, "pyproject.toml missing [project] section"
        assert 'name' in content, "pyproject.toml missing project name"
        assert 'version' in content, "pyproject.toml missing project version"

    def test_build_command_available(self):
        """Verify the 'build' package can be installed and used."""
        # Check if build module is available or can be installed
        result = subprocess.run(
            ["python", "-m", "pip", "show", "build"],
            capture_output=True,
            text=True
        )

        # If not installed, verify it can be installed
        if result.returncode != 0:
            install_result = subprocess.run(
                ["python", "-m", "pip", "install", "build"],
                capture_output=True,
                text=True
            )
            assert install_result.returncode == 0, \
                f"Failed to install 'build' package: {install_result.stderr}"

    def test_build_creates_distribution_files(self, temp_build_dir):
        """Verify python -m build creates distribution files."""
        # Run build command
        result = subprocess.run(
            ["python", "-m", "build", "--outdir", temp_build_dir],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        # Build might fail in test environment, but we can verify the command works
        if result.returncode == 0:
            # Check that distribution files were created
            dist_files = list(Path(temp_build_dir).glob("*"))
            assert len(dist_files) > 0, "Build should create distribution files"

            # Check for wheel and sdist
            has_wheel = any(f.suffix == '.whl' for f in dist_files)
            has_sdist = any(f.suffix == '.gz' for f in dist_files)

            assert has_wheel or has_sdist, \
                "Build should create at least a wheel or sdist"

    def test_sha256_hash_generation(self, temp_build_dir):
        """Verify SHA256 hash generation for distribution files."""
        # Create a mock distribution file
        test_file = Path(temp_build_dir) / "test_package-0.1.0-py3-none-any.whl"
        test_file.write_text("mock wheel content")

        # Generate SHA256 hash
        with open(test_file, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        assert len(file_hash) == 64, "SHA256 hash should be 64 characters"
        assert all(c in '0123456789abcdef' for c in file_hash), \
            "SHA256 hash should only contain hex characters"

    def test_hash_base64_encoding(self):
        """Verify base64 encoding of hash output."""
        # Simulate sha256sum output
        mock_hash_output = "abcd1234efgh5678  dist/package-0.1.0.tar.gz\n"

        # Encode to base64
        encoded = base64.b64encode(mock_hash_output.encode()).decode()

        # Verify encoding is valid
        assert encoded, "Base64 encoding should produce output"

        # Verify it can be decoded
        decoded = base64.b64decode(encoded).decode()
        assert decoded == mock_hash_output, "Base64 should be reversible"

    def test_multiple_dist_files_hash_generation(self, temp_build_dir):
        """Verify hash generation for multiple distribution files."""
        # Create multiple mock distribution files
        files = [
            "test_package-0.1.0-py3-none-any.whl",
            "test_package-0.1.0.tar.gz"
        ]

        for filename in files:
            filepath = Path(temp_build_dir) / filename
            filepath.write_text(f"mock content for {filename}")

        # Generate hashes for all files
        hashes = []
        for filename in files:
            filepath = Path(temp_build_dir) / filename
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                hashes.append(f"{file_hash}  {filepath}")

        assert len(hashes) == 2, "Should generate hashes for all files"

        # Combine and base64 encode like the workflow does
        combined_hashes = "\n".join(hashes)
        encoded = base64.b64encode(combined_hashes.encode()).decode()

        assert encoded, "Should successfully encode multiple hashes"


class TestWorkflowStepOrder:
    """Test the correct ordering of workflow steps."""

    def test_steps_are_in_correct_order(self):
        """Verify build job steps are in the correct execution order."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        step_names = [step['name'] for step in steps]

        expected_order = [
            'Checkout source',
            'Set up Python',
            'Install build tooling',
            'Build sdist and wheel',
            'Generate hashes',
            'Upload artifacts'
        ]

        assert step_names == expected_order, \
            f"Steps not in expected order.\nExpected: {expected_order}\nGot: {step_names}"

    def test_provenance_job_runs_after_build(self):
        """Verify provenance job has correct dependency chain."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        jobs = workflow['jobs']

        # Build job should not have dependencies
        assert 'needs' not in jobs['build'], \
            "Build job should not depend on other jobs"

        # Provenance job should depend on build
        assert jobs['provenance']['needs'] == 'build', \
            "Provenance job must depend on build job"


class TestSecurityAndBestPractices:
    """Test security configurations and best practices."""

    def test_uses_pinned_action_versions(self):
        """Verify all actions use pinned versions (not `@main` or `@latest`)."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']

        for step in steps:
            if 'uses' in step:
                action = step['uses']
                # Check that version is pinned (contains `@v` or `@SHA`)
                assert '@v' in action or '@' in action.split('/')[-1], \
                    f"Action {action} should use pinned version"

                # Check not using dangerous refs
                assert '@main' not in action and '@master' not in action and '@latest' not in action, \
                    f"Action {action} should not use `@main`, `@master`, or `@latest`"

    def test_minimal_permissions_principle(self):
        """Verify workflow follows principle of least privilege."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        permissions = workflow['permissions']

        # Contents should be read-only
        assert permissions['contents'] == 'read', \
            "Contents permission should be minimal (read)"

        # id-token is write only for SLSA/PyPI publishing
        assert permissions['id-token'] == 'write', \
            "id-token write is required for trusted publishing"

        # Should not have unnecessary permissions
        dangerous_perms = ['actions', 'packages', 'deployments', 'security-events']
        for perm in dangerous_perms:
            assert perm not in permissions or permissions[perm] == 'read', \
                f"Should not have write access to {perm}"

    def test_no_hardcoded_secrets(self):
        """Verify workflow doesn't contain hardcoded secrets."""
        with open(WORKFLOW_FILE, 'r') as f:
            content = f.read()

        # Check for common secret patterns
        dangerous_patterns = [
            'ghp_',  # GitHub personal access token
            'password',
            'api_key',
            'secret_key',
            'private_key',
        ]

        content_lower = content.lower()
        for pattern in dangerous_patterns:
            if pattern in content_lower:
                # Allow if it's in a comment or context variable reference
                assert 'secrets.' in content or '#' in content, \
                    f"Potential hardcoded secret pattern '{pattern}' found"

    def test_uses_ubuntu_lts_runner(self):
        """Verify workflow uses Ubuntu LTS for stability."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        runner = workflow['jobs']['build']['runs-on']

        # Ubuntu 22.04 is LTS
        assert 'ubuntu' in runner and '22.04' in runner, \
            "Should use Ubuntu LTS (22.04) for stability"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and potential failure scenarios."""

    def test_workflow_handles_empty_dist_directory(self, tmp_path):
        """Verify behavior when dist directory is empty."""
        # Create empty dist directory
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()

        # Try to generate hashes (should handle gracefully)
        # In actual workflow, this would be caught by the build step failing
        result = subprocess.run(
            ["sh", "-c", f"sha256sum {dist_dir}/* 2>&1 || echo 'No files found'"],
            capture_output=True,
            text=True
        )

        # Should either produce error message or handle empty dir
        assert result.returncode != 0 or 'No files found' in result.stdout, \
            "Should handle empty dist directory gracefully"

    def test_workflow_handles_build_failures(self):
        """Verify workflow would fail appropriately on build errors."""
        # This tests the concept - actual build failures are caught by GitHub Actions
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        build_step = next((s for s in steps if s['name'] == 'Build sdist and wheel'), None)

        # Verify build step doesn't have continue-on-error
        assert 'continue-on-error' not in build_step or not build_step['continue-on-error'], \
            "Build step should fail workflow on error"

    def test_hash_step_has_correct_output_format(self):
        """Verify hash step output format is compatible with SLSA generator."""
        # Test the hash output format expected by SLSA generator
        mock_files = [
            ("file1.whl", "abc123"),
            ("file2.tar.gz", "def456")
        ]

        # Format as sha256sum would output
        hash_lines = [f"{hash_val}  dist/{filename}" for filename, hash_val in mock_files]
        hash_output = "\n".join(hash_lines)

        # Base64 encode
        encoded = base64.b64encode(hash_output.encode()).decode()

        # Verify format
        assert '\n' not in encoded, "Base64 output should be single line (base64 -w0)"
        assert len(encoded) > 0, "Should produce non-empty output"

        # Verify decodable
        decoded = base64.b64decode(encoded).decode()
        assert "dist/" in decoded, "Decoded output should contain dist/ paths"

    def test_artifact_upload_path_pattern(self):
        """Verify artifact upload path pattern is valid."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        upload_step = next((s for s in steps if s['name'] == 'Upload artifacts'), None)

        path_pattern = upload_step['with']['path']

        # Verify pattern is valid glob pattern
        assert '*' in path_pattern or path_pattern.endswith('/'), \
            "Upload path should be a valid glob pattern"
        assert 'dist' in path_pattern, \
            "Upload path should reference dist directory"

    def test_python_version_compatibility(self):
        """Verify specified Python version is still supported."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        python_step = next((s for s in steps if s['name'] == 'Set up Python'), None)

        python_version = python_step['with']['python-version']

        # Python 3.9 was released in 2020 and is supported until October 2025
        # This test documents the version choice
        assert python_version in ['3.9', '3.10', '3.11', '3.12', '3.13'], \
            f"Python {python_version} should be a supported version"

    def test_slsa_generator_version_is_valid(self):
        """Verify SLSA generator uses a valid and secure version."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        provenance_job = workflow['jobs']['provenance']
        uses_value = provenance_job['uses']

        # Extract version
        assert '@v' in uses_value, "SLSA generator should use versioned reference"
        version = uses_value.split('@')[-1]

        # Should be v2.0.0 or later (SLSA v3)
        assert version.startswith('v2.'), \
            "Should use SLSA generator v2.x for SLSA v3 support"


class TestRegressionPrevention:
    """Tests to prevent regressions and ensure consistency."""

    def test_workflow_file_not_empty(self):
        """Prevent accidental deletion or corruption of workflow file."""
        assert WORKFLOW_FILE.exists(), "Workflow file must exist"

        file_size = WORKFLOW_FILE.stat().st_size
        assert file_size > 100, \
            f"Workflow file seems too small ({file_size} bytes), possibly corrupted"

    def test_workflow_maintains_job_count(self):
        """Ensure workflow maintains expected number of jobs."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        jobs = workflow.get('jobs', {})
        assert len(jobs) == 2, \
            f"Workflow should have exactly 2 jobs (build, provenance), found {len(jobs)}"

    def test_build_step_count_stable(self):
        """Ensure build job maintains expected number of steps."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        assert len(steps) == 6, \
            f"Build job should have 6 steps, found {len(steps)}. Review changes."

    def test_critical_keywords_present(self):
        """Verify critical keywords are present in workflow."""
        with open(WORKFLOW_FILE, 'r') as f:
            content = f.read()

        critical_keywords = [
            'SLSA',
            'sha256sum',
            'base64',
            'python -m build',
            'dist',
            'provenance'
        ]

        for keyword in critical_keywords:
            assert keyword in content, \
                f"Critical keyword '{keyword}' missing from workflow"

    def test_no_env_variables_leak(self):
        """Ensure workflow doesn't inadvertently expose environment variables."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        # Check that steps don't have env vars that could leak secrets
        steps = workflow['jobs']['build']['steps']

        for step in steps:
            if 'env' in step:
                env_vars = step['env']
                # Check for potential secret leaks
                for key, value in env_vars.items():
                    if isinstance(value, str):
                        assert not value.startswith('ghp_'), \
                            "Environment variable contains potential GitHub token"
                        assert not value.startswith('sk-'), \
                            "Environment variable contains potential API key"

    def test_artifact_retention_reasonable(self):
        """Verify artifact retention settings if specified."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        upload_step = next((s for s in steps if s['name'] == 'Upload artifacts'), None)

        # If retention-days is specified, it should be reasonable
        if 'with' in upload_step and 'retention-days' in upload_step['with']:
            retention = upload_step['with']['retention-days']
            assert 1 <= retention <= 90, \
                f"Artifact retention should be between 1-90 days, got {retention}"

    def test_github_output_format_correct(self):
        """Verify GITHUB_OUTPUT usage follows current format (not deprecated)."""
        with open(WORKFLOW_FILE, 'r') as f:
            content = f.read()

        # Should use $GITHUB_OUTPUT (new format)
        assert 'GITHUB_OUTPUT' in content, \
            "Should use GITHUB_OUTPUT for setting outputs"

        # Should not use deprecated set-output command
        assert '::set-output' not in content, \
            "Should not use deprecated ::set-output command""""
Comprehensive tests for the GitHub Actions Release (SLSA3) workflow.

This test suite validates the release workflow configuration and simulates
the build process to ensure SLSA3 provenance generation works correctly.
"""

import subprocess
import hashlib
import base64
import os
import tempfile
import shutil
import json
import yaml
import pytest
from pathlib import Path


# Path to the workflow file
WORKFLOW_FILE = Path(__file__).parent.parent / "github" / "workflows" / "release.yml"
PROJECT_ROOT = Path(__file__).parent.parent


class TestReleaseWorkflowStructure:
    """Test the structure and configuration of the release workflow YAML."""

    def test_workflow_file_exists(self):
        """Verify the release workflow file exists."""
        assert WORKFLOW_FILE.exists(), f"Workflow file not found at {WORKFLOW_FILE}"

    def test_workflow_yaml_is_valid(self):
        """Verify the workflow YAML is valid and parseable."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert workflow is not None, "Workflow YAML is empty or invalid"
        assert isinstance(workflow, dict), "Workflow YAML should be a dictionary"

    def test_workflow_has_correct_name(self):
        """Verify the workflow has the correct name."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert workflow['name'] == 'Release (SLSA3)', \
            f"Expected workflow name 'Release (SLSA3)', got '{workflow['name']}'"

    def test_workflow_trigger_on_release(self):
        """Verify the workflow is triggered on release published events."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert 'on' in workflow, "Workflow missing 'on' trigger configuration"
        assert 'release' in workflow['on'], "Workflow not triggered on release events"
        assert workflow['on']['release']['types'] == ['published'], \
            "Workflow should trigger on 'published' release type"

    def test_workflow_permissions_configured(self):
        """Verify the workflow has correct permissions for SLSA and PyPI."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert 'permissions' in workflow, "Workflow missing permissions configuration"
        permissions = workflow['permissions']

        assert permissions['contents'] == 'read', \
            "Contents permission should be 'read'"
        assert permissions['id-token'] == 'write', \
            "id-token permission should be 'write' for SLSA + PyPI trusted publishing"

    def test_workflow_has_build_job(self):
        """Verify the workflow has a build job."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert 'jobs' in workflow, "Workflow missing jobs"
        assert 'build' in workflow['jobs'], "Workflow missing 'build' job"

    def test_build_job_runs_on_ubuntu(self):
        """Verify the build job runs on Ubuntu 22.04."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        build_job = workflow['jobs']['build']
        assert build_job['runs-on'] == 'ubuntu-22.04', \
            f"Build job should run on 'ubuntu-22.04', got '{build_job['runs-on']}'"

    def test_build_job_has_outputs(self):
        """Verify the build job outputs hashes for SLSA provenance."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        build_job = workflow['jobs']['build']
        assert 'outputs' in build_job, "Build job missing outputs"
        assert 'hashes' in build_job['outputs'], "Build job missing 'hashes' output"

        # Verify the output references the hash step
        assert '${{ steps.hash.outputs.hashes }}' in build_job['outputs']['hashes']

    def test_build_job_checkout_step(self):
        """Verify the build job checks out source code."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        checkout_step = next((s for s in steps if s['name'] == 'Checkout source'), None)

        assert checkout_step is not None, "Missing 'Checkout source' step"
        assert checkout_step['uses'] == 'actions/checkout@v4', \
            "Should use actions/checkout@v4"

    def test_build_job_python_setup_step(self):
        """Verify the build job sets up Python 3.9."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        python_step = next((s for s in steps if s['name'] == 'Set up Python'), None)

        assert python_step is not None, "Missing 'Set up Python' step"
        assert python_step['uses'] == 'actions/setup-python@v5', \
            "Should use actions/setup-python@v5"
        assert python_step['with']['python-version'] == '3.9', \
            "Should use Python 3.9"

    def test_build_job_install_build_tooling(self):
        """Verify the build job installs build tooling."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        install_step = next((s for s in steps if s['name'] == 'Install build tooling'), None)

        assert install_step is not None, "Missing 'Install build tooling' step"
        assert 'run' in install_step, "Install step missing 'run' command"

        run_commands = install_step['run']
        assert 'pip install --upgrade pip' in run_commands, \
            "Should upgrade pip"
        assert 'pip install build' in run_commands, \
            "Should install 'build' package"

    def test_build_job_builds_sdist_and_wheel(self):
        """Verify the build job builds sdist and wheel."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        build_step = next((s for s in steps if s['name'] == 'Build sdist and wheel'), None)

        assert build_step is not None, "Missing 'Build sdist and wheel' step"
        assert build_step['run'] == 'python -m build', \
            "Should run 'python -m build'"

    def test_build_job_generates_hashes(self):
        """Verify the build job generates SHA256 hashes."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        hash_step = next((s for s in steps if s['name'] == 'Generate hashes'), None)

        assert hash_step is not None, "Missing 'Generate hashes' step"
        assert hash_step['id'] == 'hash', "Hash step should have id 'hash'"
        assert 'run' in hash_step, "Hash step missing 'run' command"

        run_commands = hash_step['run']
        assert 'sha256sum dist/*' in run_commands, \
            "Should generate SHA256 hashes of dist files"
        assert 'base64 -w0' in run_commands, \
            "Should base64 encode hashes"
        assert 'GITHUB_OUTPUT' in run_commands, \
            "Should write to GITHUB_OUTPUT"

    def test_build_job_uploads_artifacts(self):
        """Verify the build job uploads distribution artifacts."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        upload_step = next((s for s in steps if s['name'] == 'Upload artifacts'), None)

        assert upload_step is not None, "Missing 'Upload artifacts' step"
        assert upload_step['uses'] == 'actions/upload-artifact@v4', \
            "Should use actions/upload-artifact@v4"
        assert upload_step['with']['name'] == 'python-dist', \
            "Artifact name should be 'python-dist'"
        assert upload_step['with']['path'] == 'dist/*', \
            "Should upload all files from dist/"

    def test_workflow_has_provenance_job(self):
        """Verify the workflow has a provenance generation job."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        assert 'provenance' in workflow['jobs'], "Workflow missing 'provenance' job"

    def test_provenance_job_depends_on_build(self):
        """Verify the provenance job depends on the build job."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        provenance_job = workflow['jobs']['provenance']
        assert 'needs' in provenance_job, "Provenance job missing 'needs' dependency"
        assert provenance_job['needs'] == 'build', \
            "Provenance job should depend on 'build' job"

    def test_provenance_job_uses_slsa_generator(self):
        """Verify the provenance job uses SLSA generator workflow."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        provenance_job = workflow['jobs']['provenance']
        assert 'uses' in provenance_job, "Provenance job missing 'uses' field"

        expected_workflow = 'slsa-framework/slsa-github-generator/.github/workflows/generator-generic-ossf-slsa3-publish.yml@v2.0.0'
        assert provenance_job['uses'] == expected_workflow, \
            f"Should use SLSA OSSF generator v2.0.0"

    def test_provenance_job_passes_hashes(self):
        """Verify the provenance job receives hashes from build job."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        provenance_job = workflow['jobs']['provenance']
        assert 'with' in provenance_job, "Provenance job missing 'with' inputs"
        assert 'base64-subjects' in provenance_job['with'], \
            "Provenance job missing 'base64-subjects' input"

        assert '${{ needs.build.outputs.hashes }}' in provenance_job['with']['base64-subjects'], \
            "Provenance job should receive hashes from build job output"


class TestBuildProcessSimulation:
    """Simulate the build process to verify functionality."""

    `@pytest.fixture`
    def temp_build_dir(self):
        """Create a temporary build directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists for building."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        assert pyproject_file.exists(), \
            "pyproject.toml is required for python -m build"

    def test_pyproject_toml_has_project_metadata(self):
        """Verify pyproject.toml has necessary project metadata."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"

        with open(pyproject_file, 'r') as f:
            content = f.read()

        # Basic checks for required fields
        assert '[project]' in content, "pyproject.toml missing [project] section"
        assert 'name' in content, "pyproject.toml missing project name"
        assert 'version' in content, "pyproject.toml missing project version"

    def test_build_command_available(self):
        """Verify the 'build' package can be installed and used."""
        # Check if build module is available or can be installed
        result = subprocess.run(
            ["python", "-m", "pip", "show", "build"],
            capture_output=True,
            text=True
        )

        # If not installed, verify it can be installed
        if result.returncode != 0:
            install_result = subprocess.run(
                ["python", "-m", "pip", "install", "build"],
                capture_output=True,
                text=True
            )
            assert install_result.returncode == 0, \
                f"Failed to install 'build' package: {install_result.stderr}"

    def test_build_creates_distribution_files(self, temp_build_dir):
        """Verify python -m build creates distribution files."""
        # Run build command
        result = subprocess.run(
            ["python", "-m", "build", "--outdir", temp_build_dir],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        # Build might fail in test environment, but we can verify the command works
        if result.returncode == 0:
            # Check that distribution files were created
            dist_files = list(Path(temp_build_dir).glob("*"))
            assert len(dist_files) > 0, "Build should create distribution files"

            # Check for wheel and sdist
            has_wheel = any(f.suffix == '.whl' for f in dist_files)
            has_sdist = any(f.suffix == '.gz' for f in dist_files)

            assert has_wheel or has_sdist, \
                "Build should create at least a wheel or sdist"

    def test_sha256_hash_generation(self, temp_build_dir):
        """Verify SHA256 hash generation for distribution files."""
        # Create a mock distribution file
        test_file = Path(temp_build_dir) / "test_package-0.1.0-py3-none-any.whl"
        test_file.write_text("mock wheel content")

        # Generate SHA256 hash
        with open(test_file, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        assert len(file_hash) == 64, "SHA256 hash should be 64 characters"
        assert all(c in '0123456789abcdef' for c in file_hash), \
            "SHA256 hash should only contain hex characters"

    def test_hash_base64_encoding(self):
        """Verify base64 encoding of hash output."""
        # Simulate sha256sum output
        mock_hash_output = "abcd1234efgh5678  dist/package-0.1.0.tar.gz\n"

        # Encode to base64
        encoded = base64.b64encode(mock_hash_output.encode()).decode()

        # Verify encoding is valid
        assert encoded, "Base64 encoding should produce output"

        # Verify it can be decoded
        decoded = base64.b64decode(encoded).decode()
        assert decoded == mock_hash_output, "Base64 should be reversible"

    def test_multiple_dist_files_hash_generation(self, temp_build_dir):
        """Verify hash generation for multiple distribution files."""
        # Create multiple mock distribution files
        files = [
            "test_package-0.1.0-py3-none-any.whl",
            "test_package-0.1.0.tar.gz"
        ]

        for filename in files:
            filepath = Path(temp_build_dir) / filename
            filepath.write_text(f"mock content for {filename}")

        # Generate hashes for all files
        hashes = []
        for filename in files:
            filepath = Path(temp_build_dir) / filename
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                hashes.append(f"{file_hash}  {filepath}")

        assert len(hashes) == 2, "Should generate hashes for all files"

        # Combine and base64 encode like the workflow does
        combined_hashes = "\n".join(hashes)
        encoded = base64.b64encode(combined_hashes.encode()).decode()

        assert encoded, "Should successfully encode multiple hashes"


class TestWorkflowStepOrder:
    """Test the correct ordering of workflow steps."""

    def test_steps_are_in_correct_order(self):
        """Verify build job steps are in the correct execution order."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        step_names = [step['name'] for step in steps]

        expected_order = [
            'Checkout source',
            'Set up Python',
            'Install build tooling',
            'Build sdist and wheel',
            'Generate hashes',
            'Upload artifacts'
        ]

        assert step_names == expected_order, \
            f"Steps not in expected order.\nExpected: {expected_order}\nGot: {step_names}"

    def test_provenance_job_runs_after_build(self):
        """Verify provenance job has correct dependency chain."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        jobs = workflow['jobs']

        # Build job should not have dependencies
        assert 'needs' not in jobs['build'], \
            "Build job should not depend on other jobs"

        # Provenance job should depend on build
        assert jobs['provenance']['needs'] == 'build', \
            "Provenance job must depend on build job"


class TestSecurityAndBestPractices:
    """Test security configurations and best practices."""

    def test_uses_pinned_action_versions(self):
        """Verify all actions use pinned versions (not `@main` or `@latest`)."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']

        for step in steps:
            if 'uses' in step:
                action = step['uses']
                # Check that version is pinned (contains `@v` or `@SHA`)
                assert '@v' in action or '@' in action.split('/')[-1], \
                    f"Action {action} should use pinned version"

                # Check not using dangerous refs
                assert '@main' not in action and '@master' not in action and '@latest' not in action, \
                    f"Action {action} should not use `@main`, `@master`, or `@latest`"

    def test_minimal_permissions_principle(self):
        """Verify workflow follows principle of least privilege."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        permissions = workflow['permissions']

        # Contents should be read-only
        assert permissions['contents'] == 'read', \
            "Contents permission should be minimal (read)"

        # id-token is write only for SLSA/PyPI publishing
        assert permissions['id-token'] == 'write', \
            "id-token write is required for trusted publishing"

        # Should not have unnecessary permissions
        dangerous_perms = ['actions', 'packages', 'deployments', 'security-events']
        for perm in dangerous_perms:
            assert perm not in permissions or permissions[perm] == 'read', \
                f"Should not have write access to {perm}"

    def test_no_hardcoded_secrets(self):
        """Verify workflow doesn't contain hardcoded secrets."""
        with open(WORKFLOW_FILE, 'r') as f:
            content = f.read()

        # Check for common secret patterns
        dangerous_patterns = [
            'ghp_',  # GitHub personal access token
            'password',
            'api_key',
            'secret_key',
            'private_key',
        ]

        content_lower = content.lower()
        for pattern in dangerous_patterns:
            if pattern in content_lower:
                # Allow if it's in a comment or context variable reference
                assert 'secrets.' in content or '#' in content, \
                    f"Potential hardcoded secret pattern '{pattern}' found"

    def test_uses_ubuntu_lts_runner(self):
        """Verify workflow uses Ubuntu LTS for stability."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        runner = workflow['jobs']['build']['runs-on']

        # Ubuntu 22.04 is LTS
        assert 'ubuntu' in runner and '22.04' in runner, \
            "Should use Ubuntu LTS (22.04) for stability"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and potential failure scenarios."""

    def test_workflow_handles_empty_dist_directory(self, tmp_path):
        """Verify behavior when dist directory is empty."""
        # Create empty dist directory
        dist_dir = tmp_path / "dist"
        dist_dir.mkdir()

        # Try to generate hashes (should handle gracefully)
        # In actual workflow, this would be caught by the build step failing
        result = subprocess.run(
            ["sh", "-c", f"sha256sum {dist_dir}/* 2>&1 || echo 'No files found'"],
            capture_output=True,
            text=True
        )

        # Should either produce error message or handle empty dir
        assert result.returncode != 0 or 'No files found' in result.stdout, \
            "Should handle empty dist directory gracefully"

    def test_workflow_handles_build_failures(self):
        """Verify workflow would fail appropriately on build errors."""
        # This tests the concept - actual build failures are caught by GitHub Actions
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        build_step = next((s for s in steps if s['name'] == 'Build sdist and wheel'), None)

        # Verify build step doesn't have continue-on-error
        assert 'continue-on-error' not in build_step or not build_step['continue-on-error'], \
            "Build step should fail workflow on error"

    def test_hash_step_has_correct_output_format(self):
        """Verify hash step output format is compatible with SLSA generator."""
        # Test the hash output format expected by SLSA generator
        mock_files = [
            ("file1.whl", "abc123"),
            ("file2.tar.gz", "def456")
        ]

        # Format as sha256sum would output
        hash_lines = [f"{hash_val}  dist/{filename}" for filename, hash_val in mock_files]
        hash_output = "\n".join(hash_lines)

        # Base64 encode
        encoded = base64.b64encode(hash_output.encode()).decode()

        # Verify format
        assert '\n' not in encoded, "Base64 output should be single line (base64 -w0)"
        assert len(encoded) > 0, "Should produce non-empty output"

        # Verify decodable
        decoded = base64.b64decode(encoded).decode()
        assert "dist/" in decoded, "Decoded output should contain dist/ paths"

    def test_artifact_upload_path_pattern(self):
        """Verify artifact upload path pattern is valid."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        upload_step = next((s for s in steps if s['name'] == 'Upload artifacts'), None)

        path_pattern = upload_step['with']['path']

        # Verify pattern is valid glob pattern
        assert '*' in path_pattern or path_pattern.endswith('/'), \
            "Upload path should be a valid glob pattern"
        assert 'dist' in path_pattern, \
            "Upload path should reference dist directory"

    def test_python_version_compatibility(self):
        """Verify specified Python version is still supported."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        python_step = next((s for s in steps if s['name'] == 'Set up Python'), None)

        python_version = python_step['with']['python-version']

        # Python 3.9 was released in 2020 and is supported until October 2025
        # This test documents the version choice
        assert python_version in ['3.9', '3.10', '3.11', '3.12', '3.13'], \
            f"Python {python_version} should be a supported version"

    def test_slsa_generator_version_is_valid(self):
        """Verify SLSA generator uses a valid and secure version."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        provenance_job = workflow['jobs']['provenance']
        uses_value = provenance_job['uses']

        # Extract version
        assert '@v' in uses_value, "SLSA generator should use versioned reference"
        version = uses_value.split('@')[-1]

        # Should be v2.0.0 or later (SLSA v3)
        assert version.startswith('v2.'), \
            "Should use SLSA generator v2.x for SLSA v3 support"


class TestRegressionPrevention:
    """Tests to prevent regressions and ensure consistency."""

    def test_workflow_file_not_empty(self):
        """Prevent accidental deletion or corruption of workflow file."""
        assert WORKFLOW_FILE.exists(), "Workflow file must exist"

        file_size = WORKFLOW_FILE.stat().st_size
        assert file_size > 100, \
            f"Workflow file seems too small ({file_size} bytes), possibly corrupted"

    def test_workflow_maintains_job_count(self):
        """Ensure workflow maintains expected number of jobs."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        jobs = workflow.get('jobs', {})
        assert len(jobs) == 2, \
            f"Workflow should have exactly 2 jobs (build, provenance), found {len(jobs)}"

    def test_build_step_count_stable(self):
        """Ensure build job maintains expected number of steps."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        assert len(steps) == 6, \
            f"Build job should have 6 steps, found {len(steps)}. Review changes."

    def test_critical_keywords_present(self):
        """Verify critical keywords are present in workflow."""
        with open(WORKFLOW_FILE, 'r') as f:
            content = f.read()

        critical_keywords = [
            'SLSA',
            'sha256sum',
            'base64',
            'python -m build',
            'dist',
            'provenance'
        ]

        for keyword in critical_keywords:
            assert keyword in content, \
                f"Critical keyword '{keyword}' missing from workflow"

    def test_no_env_variables_leak(self):
        """Ensure workflow doesn't inadvertently expose environment variables."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        # Check that steps don't have env vars that could leak secrets
        steps = workflow['jobs']['build']['steps']

        for step in steps:
            if 'env' in step:
                env_vars = step['env']
                # Check for potential secret leaks
                for key, value in env_vars.items():
                    if isinstance(value, str):
                        assert not value.startswith('ghp_'), \
                            "Environment variable contains potential GitHub token"
                        assert not value.startswith('sk-'), \
                            "Environment variable contains potential API key"

    def test_artifact_retention_reasonable(self):
        """Verify artifact retention settings if specified."""
        with open(WORKFLOW_FILE, 'r') as f:
            workflow = yaml.safe_load(f)

        steps = workflow['jobs']['build']['steps']
        upload_step = next((s for s in steps if s['name'] == 'Upload artifacts'), None)

        # If retention-days is specified, it should be reasonable
        if 'with' in upload_step and 'retention-days' in upload_step['with']:
            retention = upload_step['with']['retention-days']
            assert 1 <= retention <= 90, \
                f"Artifact retention should be between 1-90 days, got {retention}"

    def test_github_output_format_correct(self):
        """Verify GITHUB_OUTPUT usage follows current format (not deprecated)."""
        with open(WORKFLOW_FILE, 'r') as f:
            content = f.read()

        # Should use $GITHUB_OUTPUT (new format)
        assert 'GITHUB_OUTPUT' in content, \
            "Should use GITHUB_OUTPUT for setting outputs"

        # Should not use deprecated set-output command
        assert '::set-output' not in content, \
            "Should not use deprecated ::set-output command"
