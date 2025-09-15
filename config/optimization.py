from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import Field

from promptx_core.config.base import BaseConfigModel
from promptx_core.config.optimizer_type import OptimizerType


class DspyOptimizationConfig(BaseConfigModel):
    """
    Base configuration class for DSPy optimization parameters.

    This class serves as the parent class for all DSPy-specific optimizer configurations,
    providing a common interface and inheritance structure for optimization settings.
    All specialized DSPy optimizers extend this base class with their specific parameters.
    """

    pass


class MIPROv2Config(DspyOptimizationConfig):
    """
    Configuration for Machine-Intelligence Prompt Optimization v2 (MIPROv2) parameters.

    MIPROv2 is an advanced prompt optimization technique that uses machine learning
    to improve prompts through iterative refinement and evaluation. This configuration
    controls the core behavior of the MIPROv2 algorithm including demo generation,
    candidate evaluation, and performance thresholds.

    Examples:
        >>> config = MIPROv2Config(
        ...     num_candidates=20,
        ...     max_bootstrapped_demos=6,
        ...     verbose=True
        ... )
    """

    auto: Optional[str] = Field(
        default="light",
        description="Automatic configuration mode - 'light' uses fewer computational "
        "resources but may be less thorough",
    )
    num_candidates: int = Field(
        default=10,
        description="Number of candidate prompts to generate during optimization",
    )
    max_bootstrapped_demos: int = Field(
        default=4,
        gt=0,
        description="Maximum number of automatically generated demonstrations "
        "to use for prompt improvement",
    )
    max_labeled_demos: int = Field(
        default=8,
        gt=0,
        description="Maximum number of human-labeled demonstrations to use for prompt improvement",
    )
    verbose: bool = Field(
        default=True,
        description="Whether to print detailed progress information during optimization",
    )
    track_stats: bool = Field(
        default=True,
        description="Whether to collect and store optimization statistics for later analysis",
    )
    num_threads: int = Field(
        default=6,
        description="Number of parallel threads to use during optimization for improved performance",
    )
    max_errors: int = Field(
        default=5,
        ge=0,
        description="Maximum number of errors allowed before optimization is aborted",
    )
    metric_threshold: Optional[float] = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Performance threshold that candidates must exceed to be considered successful",
    )
    seed: int = Field(
        default=9,
        description="Random seed for reproducible optimization results",
    )


class MIPROv2CompileConfig(DspyOptimizationConfig):
    """
    Configuration for the compilation settings used in Machine-Intelligence Prompt Optimization v2 (MIPROv2).

    This class specifies how optimization trials are executed in MIPROv2, including the number
    of trials, minibatch processing options, and proposer behaviors. It fine-tunes the optimization
    process by adjusting these parameters to achieve better performance or resource utilization.

    Examples:
        >>> compile_config = MIPROv2CompileConfig(
        ...     num_trials=50,
        ...     minibatch=True,
        ...     minibatch_size=10
        ... )
    """

    num_trials: int = Field(
        default=30,
        description="Number of optimization trials to run before selecting the best prompt",
    )
    minibatch: bool = Field(
        default=False,
        description="Whether to use minibatch processing for optimization",
    )
    minibatch_size: int = Field(
        default=25,
        description="Size of minibatches when minibatch processing is enabled",
    )
    minibatch_full_eval_steps: int = Field(
        default=10,
        description="Number of steps between full evaluations when using minibatch processing",
    )
    program_aware_proposer: bool = Field(
        default=True,
        description="Whether the proposer should use program structure to guide prompt generation",
    )
    data_aware_proposer: bool = Field(
        default=True,
        description="Whether the proposer should analyze training data to guide prompt generation",
    )
    view_data_batch_size: int = Field(
        default=10,
        description="Number of data examples to examine when data_aware_proposer is enabled",
    )
    tip_aware_proposer: bool = Field(
        default=True,
        description="Whether the proposer should use optimization tips to guide prompt generation",
    )
    fewshot_aware_proposer: bool = Field(
        default=True,
        description="Whether the proposer should utilize few-shot examples in prompt generation",
    )
    requires_permission_to_run: bool = Field(
        default=False,
        description="Whether user confirmation is required before starting optimization",
    )


class EnhancedMiPROv2Config(MIPROv2Config):
    """
    Enhanced configuration for Machine-Intelligence Prompt Optimization v2 (MIPROv2)
    with additional model settings.

    This configuration extends the basic MIPROv2Config by allowing the specification
    of custom models for prompt generation and task execution, as well as detailed
    teacher model settings for guided optimization. It is designed for advanced users
    who need more control over the modeling components used in the optimization process.

    Examples:
        >>> enhanced_config = EnhancedMiPROv2Config(
        ...     prompt_model=my_prompt_model,
        ...     task_model=my_task_model,
        ...     teacher_settings={"temperature": 0.7},
        ...     verbose=True
        ... )
    """

    prompt_model: Optional[Any] = Field(
        default=None,
        description="Custom model to use for generating optimization prompts, if different "
        "from the main model",
    )
    task_model: Optional[Any] = Field(
        default=None,
        description="Custom model to use for the actual task execution, if different from the prompt model",
    )
    teacher_settings: Dict = Field(
        default_factory=dict,
        description="Configuration settings for the teacher model used in optimization",
    )
    teacher: Any = Field(
        default=None,
        description="Teacher model instance that guides the optimization process",
    )


class MiproOptimizationConfig(BaseConfigModel):
    """Configuration for optimization parameters"""

    optimizer_type: Literal[OptimizerType.MIPROv2] = Field(
        default=OptimizerType.MIPROv2,
        description="Specifies MIPROv2 as the optimizer type - uses machine "
        "intelligence for prompt optimization and refinement",
    )
    params: MIPROv2Config = Field(
        default=MIPROv2Config(),
        description="MIPROv2-specific configuration parameters that control "
        "the optimization process and behavior",
    )
    compile: MIPROv2CompileConfig = Field(
        default=MIPROv2CompileConfig(),
        description="Compilation settings specific to MIPROv2 that determine "
        "how optimization trials are executed",
    )


class BootstrapFewShotWithOptunaConfig(DspyOptimizationConfig):
    """
    Configuration for the Bootstrap Few-Shot with Optuna optimization approach.

    This configuration controls the parameters for using Optuna to optimize few-shot prompt learning.
    It includes settings for the number of demos, Optuna rounds, candidate program generation, and
    parallel processing threads. This approach is useful for efficiently searching the hyperparameter
    space and finding effective prompt configurations with minimal manual intervention.

    Examples:
        >>> optuna_config = BootstrapFewShotWithOptunaConfig(
        ...     max_labeled_demos=10,
        ...     max_rounds=3,
        ...     num_candidate_programs=8
        ... )
    """

    max_bootstrapped_demos: int = Field(
        default=4,
        gt=0,
        description="Maximum number of automatically generated demonstrations "
        "to use in the Optuna optimization process",
    )
    max_labeled_demos: int = Field(
        default=16,
        gt=0,
        description="Maximum number of human-labeled demonstrations "
        "to incorporate during Optuna-based optimization",
    )
    max_rounds: int = Field(
        default=1,
        gt=0,
        description="Maximum number of Optuna optimization rounds "
        "to perform before selecting the best prompt",
    )
    num_candidate_programs: int = Field(
        default=16,
        gt=0,
        description="Number of candidate prompt programs to generate "
        "and evaluate during Optuna hyperparameter search",
    )
    num_threads: int = Field(
        default=6,
        gt=0,
        description="Number of parallel threads to use for concurrent "
        "evaluation of candidate prompts in Optuna",
    )


class BootstrapFewShotWithOptunaCompileConfig(DspyOptimizationConfig):
    """
    Compilation configuration for the Bootstrap Few-Shot with Optuna optimization approach.

    This configuration controls how the compiled program is generated during Optuna-based
    optimization. It specifies the maximum number of demonstrations to include in the final
    compiled program, which affects the prompt's few-shot learning capabilities.

    Examples:
        >>> compile_config = BootstrapFewShotWithOptunaCompileConfig(
        ...     max_demos=6
        ... )
    """

    max_demos: int = Field(
        default=4,
        gt=0,
        description="Maximum number of demonstrations to include "
        "in the compiled program during Optuna-based optimization",
    )


class BFSOptunaOptimizationConfig(BaseConfigModel):
    """
    Configuration wrapper for Bootstrap Few-Shot with Optuna optimization.

    This class serves as a container for the Optuna-based optimization settings,
    bundling together the optimizer type identifier, parameters configuration, and
    compilation settings. It provides a unified interface for configuring the entire
    Optuna optimization process for few-shot prompt learning.

    Examples:
        >>> optuna_config = BFSOptunaOptimizationConfig(
        ...     params=BootstrapFewShotWithOptunaConfig(max_rounds=2),
        ...     compile=BootstrapFewShotWithOptunaCompileConfig(max_demos=6)
        ... )
    """

    optimizer_type: Literal[OptimizerType.BootstrapFewShotWithOptuna] = Field(
        default=OptimizerType.BootstrapFewShotWithOptuna,
        description="Specifies Bootstrap Few-Shot with Optuna as the optimizer "
        "type - uses Optuna for hyperparameter optimization",
    )
    params: BootstrapFewShotWithOptunaConfig = Field(
        default=BootstrapFewShotWithOptunaConfig(),
        description="Configuration parameters specific to Bootstrap Few-Shot "
        "with Optuna optimization",
    )
    compile: BootstrapFewShotWithOptunaCompileConfig = Field(
        default=BootstrapFewShotWithOptunaCompileConfig(),
        description="Compilation settings for Bootstrap Few-Shot with Optuna "
        "that control how examples are selected and evaluated",
    )


class BootstrapFewShotWithRandomSearchConfig(DspyOptimizationConfig):
    """
    Configuration for the Bootstrap Few-Shot with Random Search optimization approach.

    This configuration controls the parameters for using random search to optimize few-shot prompt learning.
    It includes settings for the number of demos, optimization rounds, candidate program generation, and
    parallel processing threads. This approach is useful for exploring the hyperparameter space and finding
    effective prompt configurations through random sampling and evaluation.

    Examples:
        >>> random_search_config = BootstrapFewShotWithRandomSearchConfig(
        ...     max_labeled_demos=10,
        ...     max_rounds=3,
        ...     num_candidate_programs=8
        ... )
    """

    max_bootstrapped_demos: int = Field(
        default=4,
        gt=0,
        description="Maximum number of automatically generated demonstrations "
        "to use in the optimization process",
    )
    max_labeled_demos: int = Field(
        default=16,
        gt=0,
        description="Maximum number of human-labeled demonstrations "
        "to incorporate during optimization",
    )
    max_rounds: int = Field(
        default=1,
        gt=0,
        description="Maximum number of optimization rounds "
        "to perform before selecting the best prompt",
    )
    num_candidate_programs: int = Field(
        default=16,
        gt=0,
        description="Number of candidate prompt programs to generate and "
        "evaluate during random search",
    )
    num_threads: int = Field(
        default=6,
        gt=0,
        description="Number of parallel threads to use for concurrent "
        "evaluation of candidate prompts",
    )
    max_errors: int = Field(
        default=5,
        ge=0,
        description="Maximum number of errors allowed during optimization "
        "before aborting the process",
    )
    stop_at_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum performance score threshold at which to stop "
        "optimization early (higher value = stricter threshold)",
    )
    metric_threshold: Optional[float] = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Performance threshold that candidates must exceed "
        "to be considered successful optimizations",
    )


class BootstrapFewShotWithRandomSearchCompileConfig(DspyOptimizationConfig):
    """
    Compilation configuration for the Bootstrap Few-Shot with Random Search optimization approach.

    This configuration controls how the compiled program is generated during random search-based
    optimization. It specifies sampling strategy options for the compilation phase, affecting
    how examples are selected for inclusion in the final optimized prompt.

    Examples:
        >>> compile_config = BootstrapFewShotWithRandomSearchCompileConfig(
        ...     labeled_sample=True
        ... )
    """

    labeled_sample: bool = Field(
        default=True,
        description="Whether to use labeled examples for sampling during "
        "the compilation phase of random search optimization",
    )


class BFSRandomSearchOptimizationConfig(BaseConfigModel):
    """Configuration for optimization parameters"""

    optimizer_type: Literal[OptimizerType.BootstrapFewShotWithRandomSearch] = Field(
        default=OptimizerType.BootstrapFewShotWithRandomSearch,
        description="Specifies Bootstrap Few-Shot with Random Search "
        "as the optimizer type - uses random search to find optimal few-shot examples",
    )
    params: BootstrapFewShotWithRandomSearchConfig = Field(
        default=BootstrapFewShotWithRandomSearchConfig(),
        description="Configuration parameters specific to Bootstrap Few-Shot "
        "swith Random Search optimization",
    )
    compile: BootstrapFewShotWithRandomSearchCompileConfig = Field(
        default=BootstrapFewShotWithRandomSearchCompileConfig(),
        description="Compilation settings for Bootstrap Few-Shot with Random "
        "Search that control how examples are sampled and evaluated",
    )


class OptimizationConfig(BaseConfigModel):
    """
    General configuration class for prompt optimization parameters.

    This class provides a unified interface for configuring any supported prompt optimization method.
    It serves as a generic container that can represent any of the specific optimization approaches
    by specifying the optimizer type and corresponding parameters. This allows for a consistent
    configuration structure across different optimization techniques while maintaining flexibility.

    The config automatically handles validation of appropriate parameter combinations based on
    the selected optimizer type. It's used as the main entry point for configuring optimization
    processes in the application.

    Examples:
        >>> # Configure MIPROv2 optimization
        >>> mipro_config = OptimizationConfig(
        ...     optimizer_type=OptimizerType.MIPROv2,
        ...     params=MIPROv2Config(num_candidates=15),
        ...     compile=MIPROv2CompileConfig(num_trials=40)
        ... )
        >>>
        >>> # Configure Bootstrap Few-Shot with Random Search
        >>> bfs_config = OptimizationConfig(
        ...     optimizer_type=OptimizerType.BootstrapFewShotWithRandomSearch,
        ...     params=BootstrapFewShotWithRandomSearchConfig(max_rounds=2),
        ...     compile=BootstrapFewShotWithRandomSearchCompileConfig()
        ... )
    """

    optimizer_type: Literal[
        OptimizerType.MIPROv2,
        OptimizerType.BootstrapFewShotWithOptuna,
        OptimizerType.BootstrapFewShotWithRandomSearch,
    ] = Field(
        default=OptimizerType.MIPROv2,
        description="The type of optimizer to use for prompt optimization - "
        "determines the approach and algorithm for improving prompts",
    )
    params: Union[
        MIPROv2Config,
        BootstrapFewShotWithOptunaConfig,
        BootstrapFewShotWithRandomSearchConfig,
    ] = Field(
        default=MIPROv2Config(),
        description="Configuration parameters specific to the selected optimizer "
        "type - controls behavior during the optimization process",
    )
    compile: Union[
        MIPROv2CompileConfig,
        BootstrapFewShotWithOptunaCompileConfig,
        BootstrapFewShotWithRandomSearchCompileConfig,
    ] = Field(
        default=MIPROv2CompileConfig(),
        description="Compilation settings for the optimizer - "
        "determines how optimization trials are executed and evaluated",
    )


"""
An annotated union type that enables automatic optimization config selection.

This type leverages Pydantic's discriminated unions to automatically deserialize
the correct optimizer configuration based on the 'optimizer_type' field.
When used in config loading, it will instantiate the appropriate optimizer class
without requiring explicit type checking in the application code.

Example:
    config = load_config_from_file("config.yaml")
    # If config.yaml contains optimizer_type: "MIPROv2", this will be
    #  a MiproOptimizationConfig
    # If config.yaml contains optimizer_type: "BootstrapFewShotWithOptuna",
    # this will be a BFSOptunaOptimizationConfig
    optimizer_config = config.optimization
"""
OptimizationConfigAnnotated = Annotated[
    Union[
        MiproOptimizationConfig,
        BFSOptunaOptimizationConfig,
        BFSRandomSearchOptimizationConfig,
    ],
    Field(
        discriminator="optimizer_type",
        description="Annotated configuration type that automatically selects the "
        "appropriate optimizer config based on the 'optimizer_type' field value",
    ),
]
