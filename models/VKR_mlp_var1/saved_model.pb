¤
Ð¦
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.22v2.6.1-9-gc2363d6d0258©
|
dense_496/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_namedense_496/kernel
u
$dense_496/kernel/Read/ReadVariableOpReadVariableOpdense_496/kernel*
_output_shapes

:
*
dtype0
t
dense_496/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_496/bias
m
"dense_496/bias/Read/ReadVariableOpReadVariableOpdense_496/bias*
_output_shapes
:*
dtype0
|
dense_497/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_497/kernel
u
$dense_497/kernel/Read/ReadVariableOpReadVariableOpdense_497/kernel*
_output_shapes

:*
dtype0
t
dense_497/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_497/bias
m
"dense_497/bias/Read/ReadVariableOpReadVariableOpdense_497/bias*
_output_shapes
:*
dtype0
|
dense_498/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_498/kernel
u
$dense_498/kernel/Read/ReadVariableOpReadVariableOpdense_498/kernel*
_output_shapes

:*
dtype0
t
dense_498/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_498/bias
m
"dense_498/bias/Read/ReadVariableOpReadVariableOpdense_498/bias*
_output_shapes
:*
dtype0
|
dense_499/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_499/kernel
u
$dense_499/kernel/Read/ReadVariableOpReadVariableOpdense_499/kernel*
_output_shapes

:*
dtype0
t
dense_499/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_499/bias
m
"dense_499/bias/Read/ReadVariableOpReadVariableOpdense_499/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

NoOpNoOp
û 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¶ 
value¬ B©  B¢ 

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
6
(iter
	)decay
*learning_rate
+momentum
8
0
1
2
3
4
5
"6
#7
 
8
0
1
2
3
4
5
"6
#7
­

,layers
trainable_variables
-layer_regularization_losses
.metrics
/layer_metrics
regularization_losses
0non_trainable_variables
		variables
 
\Z
VARIABLE_VALUEdense_496/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_496/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

1layers
trainable_variables
2layer_regularization_losses
3metrics
4layer_metrics
regularization_losses
5non_trainable_variables
	variables
\Z
VARIABLE_VALUEdense_497/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_497/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

6layers
trainable_variables
7layer_regularization_losses
8metrics
9layer_metrics
regularization_losses
:non_trainable_variables
	variables
\Z
VARIABLE_VALUEdense_498/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_498/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

;layers
trainable_variables
<layer_regularization_losses
=metrics
>layer_metrics
regularization_losses
?non_trainable_variables
	variables
 
 
 
­

@layers
trainable_variables
Alayer_regularization_losses
Bmetrics
Clayer_metrics
regularization_losses
Dnon_trainable_variables
 	variables
\Z
VARIABLE_VALUEdense_499/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_499/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
­

Elayers
$trainable_variables
Flayer_regularization_losses
Gmetrics
Hlayer_metrics
%regularization_losses
Inon_trainable_variables
&	variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
2
3
4
 

J0
K1
L2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Mtotal
	Ncount
O	variables
P	keras_api
D
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api
D
	Vtotal
	Wcount
X
_fn_kwargs
Y	variables
Z	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

O	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

T	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

Y	variables

serving_default_dense_496_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

Ó
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_496_inputdense_496/kerneldense_496/biasdense_497/kerneldense_497/biasdense_498/kerneldense_498/biasdense_499/kerneldense_499/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_6716882
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_496/kernel/Read/ReadVariableOp"dense_496/bias/Read/ReadVariableOp$dense_497/kernel/Read/ReadVariableOp"dense_497/bias/Read/ReadVariableOp$dense_498/kernel/Read/ReadVariableOp"dense_498/bias/Read/ReadVariableOp$dense_499/kernel/Read/ReadVariableOp"dense_499/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_6717181

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_496/kerneldense_496/biasdense_497/kerneldense_497/biasdense_498/kerneldense_498/biasdense_499/kerneldense_499/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1total_2count_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_6717245ùØ

÷
F__inference_dense_496_layer_call_and_return_conditional_losses_6716570

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
õ
f
H__inference_dropout_196_layer_call_and_return_conditional_losses_6717062

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
g
H__inference_dropout_196_layer_call_and_return_conditional_losses_6716684

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
f
H__inference_dropout_196_layer_call_and_return_conditional_losses_6716615

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú

÷
F__inference_dense_499_layer_call_and_return_conditional_losses_6716628

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
g
H__inference_dropout_196_layer_call_and_return_conditional_losses_6717074

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÀ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2
dropout/GreaterEqual/y¾
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³M
ú	
#__inference__traced_restore_6717245
file_prefix3
!assignvariableop_dense_496_kernel:
/
!assignvariableop_1_dense_496_bias:5
#assignvariableop_2_dense_497_kernel:/
!assignvariableop_3_dense_497_bias:5
#assignvariableop_4_dense_498_kernel:/
!assignvariableop_5_dense_498_bias:5
#assignvariableop_6_dense_499_kernel:/
!assignvariableop_7_dense_499_bias:%
assignvariableop_8_sgd_iter:	 &
assignvariableop_9_sgd_decay: /
%assignvariableop_10_sgd_learning_rate: *
 assignvariableop_11_sgd_momentum: #
assignvariableop_12_total: #
assignvariableop_13_count: %
assignvariableop_14_total_1: %
assignvariableop_15_count_1: %
assignvariableop_16_total_2: %
assignvariableop_17_count_2: 
identity_19¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ý
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*é
valueßBÜB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_dense_496_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_496_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_497_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_497_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_498_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_498_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_499_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_499_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8 
AssignVariableOp_8AssignVariableOpassignvariableop_8_sgd_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¡
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10­
AssignVariableOp_10AssignVariableOp%assignvariableop_10_sgd_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¨
AssignVariableOp_11AssignVariableOp assignvariableop_11_sgd_momentumIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¡
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¡
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14£
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15£
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16£
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpê
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18f
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_19Ò
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
û	
¿
0__inference_sequential_196_layer_call_fn_6716997

inputs
unknown:

	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_196_layer_call_and_return_conditional_losses_67167652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

¸
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716765

inputs#
dense_496_6716743:

dense_496_6716745:#
dense_497_6716748:
dense_497_6716750:#
dense_498_6716753:
dense_498_6716755:#
dense_499_6716759:
dense_499_6716761:
identity¢!dense_496/StatefulPartitionedCall¢!dense_497/StatefulPartitionedCall¢!dense_498/StatefulPartitionedCall¢!dense_499/StatefulPartitionedCall¢#dropout_196/StatefulPartitionedCall
!dense_496/StatefulPartitionedCallStatefulPartitionedCallinputsdense_496_6716743dense_496_6716745*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_496_layer_call_and_return_conditional_losses_67165702#
!dense_496/StatefulPartitionedCallÀ
!dense_497/StatefulPartitionedCallStatefulPartitionedCall*dense_496/StatefulPartitionedCall:output:0dense_497_6716748dense_497_6716750*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_497_layer_call_and_return_conditional_losses_67165872#
!dense_497/StatefulPartitionedCallÀ
!dense_498/StatefulPartitionedCallStatefulPartitionedCall*dense_497/StatefulPartitionedCall:output:0dense_498_6716753dense_498_6716755*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_498_layer_call_and_return_conditional_losses_67166042#
!dense_498/StatefulPartitionedCall
#dropout_196/StatefulPartitionedCallStatefulPartitionedCall*dense_498/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_196_layer_call_and_return_conditional_losses_67166842%
#dropout_196/StatefulPartitionedCallÂ
!dense_499/StatefulPartitionedCallStatefulPartitionedCall,dropout_196/StatefulPartitionedCall:output:0dense_499_6716759dense_499_6716761*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_499_layer_call_and_return_conditional_losses_67166282#
!dense_499/StatefulPartitionedCall
IdentityIdentity*dense_499/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp"^dense_496/StatefulPartitionedCall"^dense_497/StatefulPartitionedCall"^dense_498/StatefulPartitionedCall"^dense_499/StatefulPartitionedCall$^dropout_196/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 2F
!dense_496/StatefulPartitionedCall!dense_496/StatefulPartitionedCall2F
!dense_497/StatefulPartitionedCall!dense_497/StatefulPartitionedCall2F
!dense_498/StatefulPartitionedCall!dense_498/StatefulPartitionedCall2F
!dense_499/StatefulPartitionedCall!dense_499/StatefulPartitionedCall2J
#dropout_196/StatefulPartitionedCall#dropout_196/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¶4
Ò
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716955

inputs:
(dense_496_matmul_readvariableop_resource:
7
)dense_496_biasadd_readvariableop_resource::
(dense_497_matmul_readvariableop_resource:7
)dense_497_biasadd_readvariableop_resource::
(dense_498_matmul_readvariableop_resource:7
)dense_498_biasadd_readvariableop_resource::
(dense_499_matmul_readvariableop_resource:7
)dense_499_biasadd_readvariableop_resource:
identity¢ dense_496/BiasAdd/ReadVariableOp¢dense_496/MatMul/ReadVariableOp¢ dense_497/BiasAdd/ReadVariableOp¢dense_497/MatMul/ReadVariableOp¢ dense_498/BiasAdd/ReadVariableOp¢dense_498/MatMul/ReadVariableOp¢ dense_499/BiasAdd/ReadVariableOp¢dense_499/MatMul/ReadVariableOp«
dense_496/MatMul/ReadVariableOpReadVariableOp(dense_496_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
dense_496/MatMul/ReadVariableOp
dense_496/MatMulMatMulinputs'dense_496/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_496/MatMulª
 dense_496/BiasAdd/ReadVariableOpReadVariableOp)dense_496_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_496/BiasAdd/ReadVariableOp©
dense_496/BiasAddBiasAdddense_496/MatMul:product:0(dense_496/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_496/BiasAdd
dense_496/SoftmaxSoftmaxdense_496/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_496/Softmax«
dense_497/MatMul/ReadVariableOpReadVariableOp(dense_497_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_497/MatMul/ReadVariableOp¦
dense_497/MatMulMatMuldense_496/Softmax:softmax:0'dense_497/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_497/MatMulª
 dense_497/BiasAdd/ReadVariableOpReadVariableOp)dense_497_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_497/BiasAdd/ReadVariableOp©
dense_497/BiasAddBiasAdddense_497/MatMul:product:0(dense_497/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_497/BiasAdd
dense_497/SoftmaxSoftmaxdense_497/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_497/Softmax«
dense_498/MatMul/ReadVariableOpReadVariableOp(dense_498_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_498/MatMul/ReadVariableOp¦
dense_498/MatMulMatMuldense_497/Softmax:softmax:0'dense_498/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_498/MatMulª
 dense_498/BiasAdd/ReadVariableOpReadVariableOp)dense_498_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_498/BiasAdd/ReadVariableOp©
dense_498/BiasAddBiasAdddense_498/MatMul:product:0(dense_498/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_498/BiasAdd
dense_498/SoftmaxSoftmaxdense_498/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_498/Softmax{
dropout_196/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¢¼?2
dropout_196/dropout/Const¬
dropout_196/dropout/MulMuldense_498/Softmax:softmax:0"dropout_196/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_196/dropout/Mul
dropout_196/dropout/ShapeShapedense_498/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout_196/dropout/Shapeä
0dropout_196/dropout/random_uniform/RandomUniformRandomUniform"dropout_196/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed22
0dropout_196/dropout/random_uniform/RandomUniform
"dropout_196/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL=2$
"dropout_196/dropout/GreaterEqual/yî
 dropout_196/dropout/GreaterEqualGreaterEqual9dropout_196/dropout/random_uniform/RandomUniform:output:0+dropout_196/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 dropout_196/dropout/GreaterEqual£
dropout_196/dropout/CastCast$dropout_196/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_196/dropout/Castª
dropout_196/dropout/Mul_1Muldropout_196/dropout/Mul:z:0dropout_196/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_196/dropout/Mul_1«
dense_499/MatMul/ReadVariableOpReadVariableOp(dense_499_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_499/MatMul/ReadVariableOp¨
dense_499/MatMulMatMuldropout_196/dropout/Mul_1:z:0'dense_499/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_499/MatMulª
 dense_499/BiasAdd/ReadVariableOpReadVariableOp)dense_499_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_499/BiasAdd/ReadVariableOp©
dense_499/BiasAddBiasAdddense_499/MatMul:product:0(dense_499/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_499/BiasAddv
dense_499/TanhTanhdense_499/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_499/Tanhm
IdentityIdentitydense_499/Tanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityâ
NoOpNoOp!^dense_496/BiasAdd/ReadVariableOp ^dense_496/MatMul/ReadVariableOp!^dense_497/BiasAdd/ReadVariableOp ^dense_497/MatMul/ReadVariableOp!^dense_498/BiasAdd/ReadVariableOp ^dense_498/MatMul/ReadVariableOp!^dense_499/BiasAdd/ReadVariableOp ^dense_499/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 2D
 dense_496/BiasAdd/ReadVariableOp dense_496/BiasAdd/ReadVariableOp2B
dense_496/MatMul/ReadVariableOpdense_496/MatMul/ReadVariableOp2D
 dense_497/BiasAdd/ReadVariableOp dense_497/BiasAdd/ReadVariableOp2B
dense_497/MatMul/ReadVariableOpdense_497/MatMul/ReadVariableOp2D
 dense_498/BiasAdd/ReadVariableOp dense_498/BiasAdd/ReadVariableOp2B
dense_498/MatMul/ReadVariableOpdense_498/MatMul/ReadVariableOp2D
 dense_499/BiasAdd/ReadVariableOp dense_499/BiasAdd/ReadVariableOp2B
dense_499/MatMul/ReadVariableOpdense_499/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


È
0__inference_sequential_196_layer_call_fn_6716805
dense_496_input
unknown:

	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCalldense_496_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_196_layer_call_and_return_conditional_losses_67167652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namedense_496_input

÷
F__inference_dense_498_layer_call_and_return_conditional_losses_6717048

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì7
¢
"__inference__wrapped_model_6716552
dense_496_inputI
7sequential_196_dense_496_matmul_readvariableop_resource:
F
8sequential_196_dense_496_biasadd_readvariableop_resource:I
7sequential_196_dense_497_matmul_readvariableop_resource:F
8sequential_196_dense_497_biasadd_readvariableop_resource:I
7sequential_196_dense_498_matmul_readvariableop_resource:F
8sequential_196_dense_498_biasadd_readvariableop_resource:I
7sequential_196_dense_499_matmul_readvariableop_resource:F
8sequential_196_dense_499_biasadd_readvariableop_resource:
identity¢/sequential_196/dense_496/BiasAdd/ReadVariableOp¢.sequential_196/dense_496/MatMul/ReadVariableOp¢/sequential_196/dense_497/BiasAdd/ReadVariableOp¢.sequential_196/dense_497/MatMul/ReadVariableOp¢/sequential_196/dense_498/BiasAdd/ReadVariableOp¢.sequential_196/dense_498/MatMul/ReadVariableOp¢/sequential_196/dense_499/BiasAdd/ReadVariableOp¢.sequential_196/dense_499/MatMul/ReadVariableOpØ
.sequential_196/dense_496/MatMul/ReadVariableOpReadVariableOp7sequential_196_dense_496_matmul_readvariableop_resource*
_output_shapes

:
*
dtype020
.sequential_196/dense_496/MatMul/ReadVariableOpÇ
sequential_196/dense_496/MatMulMatMuldense_496_input6sequential_196/dense_496/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_196/dense_496/MatMul×
/sequential_196/dense_496/BiasAdd/ReadVariableOpReadVariableOp8sequential_196_dense_496_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_196/dense_496/BiasAdd/ReadVariableOpå
 sequential_196/dense_496/BiasAddBiasAdd)sequential_196/dense_496/MatMul:product:07sequential_196/dense_496/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_196/dense_496/BiasAdd¬
 sequential_196/dense_496/SoftmaxSoftmax)sequential_196/dense_496/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_196/dense_496/SoftmaxØ
.sequential_196/dense_497/MatMul/ReadVariableOpReadVariableOp7sequential_196_dense_497_matmul_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_196/dense_497/MatMul/ReadVariableOpâ
sequential_196/dense_497/MatMulMatMul*sequential_196/dense_496/Softmax:softmax:06sequential_196/dense_497/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_196/dense_497/MatMul×
/sequential_196/dense_497/BiasAdd/ReadVariableOpReadVariableOp8sequential_196_dense_497_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_196/dense_497/BiasAdd/ReadVariableOpå
 sequential_196/dense_497/BiasAddBiasAdd)sequential_196/dense_497/MatMul:product:07sequential_196/dense_497/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_196/dense_497/BiasAdd¬
 sequential_196/dense_497/SoftmaxSoftmax)sequential_196/dense_497/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_196/dense_497/SoftmaxØ
.sequential_196/dense_498/MatMul/ReadVariableOpReadVariableOp7sequential_196_dense_498_matmul_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_196/dense_498/MatMul/ReadVariableOpâ
sequential_196/dense_498/MatMulMatMul*sequential_196/dense_497/Softmax:softmax:06sequential_196/dense_498/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_196/dense_498/MatMul×
/sequential_196/dense_498/BiasAdd/ReadVariableOpReadVariableOp8sequential_196_dense_498_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_196/dense_498/BiasAdd/ReadVariableOpå
 sequential_196/dense_498/BiasAddBiasAdd)sequential_196/dense_498/MatMul:product:07sequential_196/dense_498/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_196/dense_498/BiasAdd¬
 sequential_196/dense_498/SoftmaxSoftmax)sequential_196/dense_498/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_196/dense_498/Softmax´
#sequential_196/dropout_196/IdentityIdentity*sequential_196/dense_498/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential_196/dropout_196/IdentityØ
.sequential_196/dense_499/MatMul/ReadVariableOpReadVariableOp7sequential_196_dense_499_matmul_readvariableop_resource*
_output_shapes

:*
dtype020
.sequential_196/dense_499/MatMul/ReadVariableOpä
sequential_196/dense_499/MatMulMatMul,sequential_196/dropout_196/Identity:output:06sequential_196/dense_499/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_196/dense_499/MatMul×
/sequential_196/dense_499/BiasAdd/ReadVariableOpReadVariableOp8sequential_196_dense_499_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_196/dense_499/BiasAdd/ReadVariableOpå
 sequential_196/dense_499/BiasAddBiasAdd)sequential_196/dense_499/MatMul:product:07sequential_196/dense_499/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_196/dense_499/BiasAdd£
sequential_196/dense_499/TanhTanh)sequential_196/dense_499/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_196/dense_499/Tanh|
IdentityIdentity!sequential_196/dense_499/Tanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÚ
NoOpNoOp0^sequential_196/dense_496/BiasAdd/ReadVariableOp/^sequential_196/dense_496/MatMul/ReadVariableOp0^sequential_196/dense_497/BiasAdd/ReadVariableOp/^sequential_196/dense_497/MatMul/ReadVariableOp0^sequential_196/dense_498/BiasAdd/ReadVariableOp/^sequential_196/dense_498/MatMul/ReadVariableOp0^sequential_196/dense_499/BiasAdd/ReadVariableOp/^sequential_196/dense_499/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 2b
/sequential_196/dense_496/BiasAdd/ReadVariableOp/sequential_196/dense_496/BiasAdd/ReadVariableOp2`
.sequential_196/dense_496/MatMul/ReadVariableOp.sequential_196/dense_496/MatMul/ReadVariableOp2b
/sequential_196/dense_497/BiasAdd/ReadVariableOp/sequential_196/dense_497/BiasAdd/ReadVariableOp2`
.sequential_196/dense_497/MatMul/ReadVariableOp.sequential_196/dense_497/MatMul/ReadVariableOp2b
/sequential_196/dense_498/BiasAdd/ReadVariableOp/sequential_196/dense_498/BiasAdd/ReadVariableOp2`
.sequential_196/dense_498/MatMul/ReadVariableOp.sequential_196/dense_498/MatMul/ReadVariableOp2b
/sequential_196/dense_499/BiasAdd/ReadVariableOp/sequential_196/dense_499/BiasAdd/ReadVariableOp2`
.sequential_196/dense_499/MatMul/ReadVariableOp.sequential_196/dense_499/MatMul/ReadVariableOp:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namedense_496_input

÷
F__inference_dense_497_layer_call_and_return_conditional_losses_6716587

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ*
Ò
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716915

inputs:
(dense_496_matmul_readvariableop_resource:
7
)dense_496_biasadd_readvariableop_resource::
(dense_497_matmul_readvariableop_resource:7
)dense_497_biasadd_readvariableop_resource::
(dense_498_matmul_readvariableop_resource:7
)dense_498_biasadd_readvariableop_resource::
(dense_499_matmul_readvariableop_resource:7
)dense_499_biasadd_readvariableop_resource:
identity¢ dense_496/BiasAdd/ReadVariableOp¢dense_496/MatMul/ReadVariableOp¢ dense_497/BiasAdd/ReadVariableOp¢dense_497/MatMul/ReadVariableOp¢ dense_498/BiasAdd/ReadVariableOp¢dense_498/MatMul/ReadVariableOp¢ dense_499/BiasAdd/ReadVariableOp¢dense_499/MatMul/ReadVariableOp«
dense_496/MatMul/ReadVariableOpReadVariableOp(dense_496_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02!
dense_496/MatMul/ReadVariableOp
dense_496/MatMulMatMulinputs'dense_496/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_496/MatMulª
 dense_496/BiasAdd/ReadVariableOpReadVariableOp)dense_496_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_496/BiasAdd/ReadVariableOp©
dense_496/BiasAddBiasAdddense_496/MatMul:product:0(dense_496/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_496/BiasAdd
dense_496/SoftmaxSoftmaxdense_496/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_496/Softmax«
dense_497/MatMul/ReadVariableOpReadVariableOp(dense_497_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_497/MatMul/ReadVariableOp¦
dense_497/MatMulMatMuldense_496/Softmax:softmax:0'dense_497/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_497/MatMulª
 dense_497/BiasAdd/ReadVariableOpReadVariableOp)dense_497_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_497/BiasAdd/ReadVariableOp©
dense_497/BiasAddBiasAdddense_497/MatMul:product:0(dense_497/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_497/BiasAdd
dense_497/SoftmaxSoftmaxdense_497/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_497/Softmax«
dense_498/MatMul/ReadVariableOpReadVariableOp(dense_498_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_498/MatMul/ReadVariableOp¦
dense_498/MatMulMatMuldense_497/Softmax:softmax:0'dense_498/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_498/MatMulª
 dense_498/BiasAdd/ReadVariableOpReadVariableOp)dense_498_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_498/BiasAdd/ReadVariableOp©
dense_498/BiasAddBiasAdddense_498/MatMul:product:0(dense_498/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_498/BiasAdd
dense_498/SoftmaxSoftmaxdense_498/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_498/Softmax
dropout_196/IdentityIdentitydense_498/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_196/Identity«
dense_499/MatMul/ReadVariableOpReadVariableOp(dense_499_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_499/MatMul/ReadVariableOp¨
dense_499/MatMulMatMuldropout_196/Identity:output:0'dense_499/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_499/MatMulª
 dense_499/BiasAdd/ReadVariableOpReadVariableOp)dense_499_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_499/BiasAdd/ReadVariableOp©
dense_499/BiasAddBiasAdddense_499/MatMul:product:0(dense_499/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_499/BiasAddv
dense_499/TanhTanhdense_499/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_499/Tanhm
IdentityIdentitydense_499/Tanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityâ
NoOpNoOp!^dense_496/BiasAdd/ReadVariableOp ^dense_496/MatMul/ReadVariableOp!^dense_497/BiasAdd/ReadVariableOp ^dense_497/MatMul/ReadVariableOp!^dense_498/BiasAdd/ReadVariableOp ^dense_498/MatMul/ReadVariableOp!^dense_499/BiasAdd/ReadVariableOp ^dense_499/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 2D
 dense_496/BiasAdd/ReadVariableOp dense_496/BiasAdd/ReadVariableOp2B
dense_496/MatMul/ReadVariableOpdense_496/MatMul/ReadVariableOp2D
 dense_497/BiasAdd/ReadVariableOp dense_497/BiasAdd/ReadVariableOp2B
dense_497/MatMul/ReadVariableOpdense_497/MatMul/ReadVariableOp2D
 dense_498/BiasAdd/ReadVariableOp dense_498/BiasAdd/ReadVariableOp2B
dense_498/MatMul/ReadVariableOpdense_498/MatMul/ReadVariableOp2D
 dense_499/BiasAdd/ReadVariableOp dense_499/BiasAdd/ReadVariableOp2B
dense_499/MatMul/ReadVariableOpdense_499/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs


È
0__inference_sequential_196_layer_call_fn_6716654
dense_496_input
unknown:

	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCalldense_496_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_196_layer_call_and_return_conditional_losses_67166352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namedense_496_input
õ

+__inference_dense_496_layer_call_fn_6717017

inputs
unknown:

	unknown_0:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_496_layer_call_and_return_conditional_losses_67165702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

÷
F__inference_dense_497_layer_call_and_return_conditional_losses_6717028

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ

+__inference_dense_498_layer_call_fn_6717057

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_498_layer_call_and_return_conditional_losses_67166042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±,

 __inference__traced_save_6717181
file_prefix/
+savev2_dense_496_kernel_read_readvariableop-
)savev2_dense_496_bias_read_readvariableop/
+savev2_dense_497_kernel_read_readvariableop-
)savev2_dense_497_bias_read_readvariableop/
+savev2_dense_498_kernel_read_readvariableop-
)savev2_dense_498_bias_read_readvariableop/
+savev2_dense_499_kernel_read_readvariableop-
)savev2_dense_499_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename×
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*é
valueßBÜB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names®
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices¢
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_496_kernel_read_readvariableop)savev2_dense_496_bias_read_readvariableop+savev2_dense_497_kernel_read_readvariableop)savev2_dense_497_bias_read_readvariableop+savev2_dense_498_kernel_read_readvariableop)savev2_dense_498_bias_read_readvariableop+savev2_dense_499_kernel_read_readvariableop)savev2_dense_499_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*k
_input_shapesZ
X: :
:::::::: : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
×

K__inference_sequential_196_layer_call_and_return_conditional_losses_6716635

inputs#
dense_496_6716571:

dense_496_6716573:#
dense_497_6716588:
dense_497_6716590:#
dense_498_6716605:
dense_498_6716607:#
dense_499_6716629:
dense_499_6716631:
identity¢!dense_496/StatefulPartitionedCall¢!dense_497/StatefulPartitionedCall¢!dense_498/StatefulPartitionedCall¢!dense_499/StatefulPartitionedCall
!dense_496/StatefulPartitionedCallStatefulPartitionedCallinputsdense_496_6716571dense_496_6716573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_496_layer_call_and_return_conditional_losses_67165702#
!dense_496/StatefulPartitionedCallÀ
!dense_497/StatefulPartitionedCallStatefulPartitionedCall*dense_496/StatefulPartitionedCall:output:0dense_497_6716588dense_497_6716590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_497_layer_call_and_return_conditional_losses_67165872#
!dense_497/StatefulPartitionedCallÀ
!dense_498/StatefulPartitionedCallStatefulPartitionedCall*dense_497/StatefulPartitionedCall:output:0dense_498_6716605dense_498_6716607*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_498_layer_call_and_return_conditional_losses_67166042#
!dense_498/StatefulPartitionedCall
dropout_196/PartitionedCallPartitionedCall*dense_498/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_196_layer_call_and_return_conditional_losses_67166152
dropout_196/PartitionedCallº
!dense_499/StatefulPartitionedCallStatefulPartitionedCall$dropout_196/PartitionedCall:output:0dense_499_6716629dense_499_6716631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_499_layer_call_and_return_conditional_losses_67166282#
!dense_499/StatefulPartitionedCall
IdentityIdentity*dense_499/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÞ
NoOpNoOp"^dense_496/StatefulPartitionedCall"^dense_497/StatefulPartitionedCall"^dense_498/StatefulPartitionedCall"^dense_499/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 2F
!dense_496/StatefulPartitionedCall!dense_496/StatefulPartitionedCall2F
!dense_497/StatefulPartitionedCall!dense_497/StatefulPartitionedCall2F
!dense_498/StatefulPartitionedCall!dense_498/StatefulPartitionedCall2F
!dense_499/StatefulPartitionedCall!dense_499/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
õ

+__inference_dense_499_layer_call_fn_6717104

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_499_layer_call_and_return_conditional_losses_67166282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò

K__inference_sequential_196_layer_call_and_return_conditional_losses_6716830
dense_496_input#
dense_496_6716808:

dense_496_6716810:#
dense_497_6716813:
dense_497_6716815:#
dense_498_6716818:
dense_498_6716820:#
dense_499_6716824:
dense_499_6716826:
identity¢!dense_496/StatefulPartitionedCall¢!dense_497/StatefulPartitionedCall¢!dense_498/StatefulPartitionedCall¢!dense_499/StatefulPartitionedCall¥
!dense_496/StatefulPartitionedCallStatefulPartitionedCalldense_496_inputdense_496_6716808dense_496_6716810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_496_layer_call_and_return_conditional_losses_67165702#
!dense_496/StatefulPartitionedCallÀ
!dense_497/StatefulPartitionedCallStatefulPartitionedCall*dense_496/StatefulPartitionedCall:output:0dense_497_6716813dense_497_6716815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_497_layer_call_and_return_conditional_losses_67165872#
!dense_497/StatefulPartitionedCallÀ
!dense_498/StatefulPartitionedCallStatefulPartitionedCall*dense_497/StatefulPartitionedCall:output:0dense_498_6716818dense_498_6716820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_498_layer_call_and_return_conditional_losses_67166042#
!dense_498/StatefulPartitionedCall
dropout_196/PartitionedCallPartitionedCall*dense_498/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_196_layer_call_and_return_conditional_losses_67166152
dropout_196/PartitionedCallº
!dense_499/StatefulPartitionedCallStatefulPartitionedCall$dropout_196/PartitionedCall:output:0dense_499_6716824dense_499_6716826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_499_layer_call_and_return_conditional_losses_67166282#
!dense_499/StatefulPartitionedCall
IdentityIdentity*dense_499/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÞ
NoOpNoOp"^dense_496/StatefulPartitionedCall"^dense_497/StatefulPartitionedCall"^dense_498/StatefulPartitionedCall"^dense_499/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 2F
!dense_496/StatefulPartitionedCall!dense_496/StatefulPartitionedCall2F
!dense_497/StatefulPartitionedCall!dense_497/StatefulPartitionedCall2F
!dense_498/StatefulPartitionedCall!dense_498/StatefulPartitionedCall2F
!dense_499/StatefulPartitionedCall!dense_499/StatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namedense_496_input

÷
F__inference_dense_496_layer_call_and_return_conditional_losses_6717008

inputs0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
â	
½
%__inference_signature_wrapper_6716882
dense_496_input
unknown:

	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCalldense_496_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_67165522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namedense_496_input

÷
F__inference_dense_498_layer_call_and_return_conditional_losses_6716604

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
f
-__inference_dropout_196_layer_call_fn_6717084

inputs
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_196_layer_call_and_return_conditional_losses_67166842
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ

+__inference_dense_497_layer_call_fn_6717037

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_497_layer_call_and_return_conditional_losses_67165872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
I
-__inference_dropout_196_layer_call_fn_6717079

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_196_layer_call_and_return_conditional_losses_67166152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú

÷
F__inference_dense_499_layer_call_and_return_conditional_losses_6717095

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanhc
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û	
¿
0__inference_sequential_196_layer_call_fn_6716976

inputs
unknown:

	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_196_layer_call_and_return_conditional_losses_67166352
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ª
Á
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716855
dense_496_input#
dense_496_6716833:

dense_496_6716835:#
dense_497_6716838:
dense_497_6716840:#
dense_498_6716843:
dense_498_6716845:#
dense_499_6716849:
dense_499_6716851:
identity¢!dense_496/StatefulPartitionedCall¢!dense_497/StatefulPartitionedCall¢!dense_498/StatefulPartitionedCall¢!dense_499/StatefulPartitionedCall¢#dropout_196/StatefulPartitionedCall¥
!dense_496/StatefulPartitionedCallStatefulPartitionedCalldense_496_inputdense_496_6716833dense_496_6716835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_496_layer_call_and_return_conditional_losses_67165702#
!dense_496/StatefulPartitionedCallÀ
!dense_497/StatefulPartitionedCallStatefulPartitionedCall*dense_496/StatefulPartitionedCall:output:0dense_497_6716838dense_497_6716840*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_497_layer_call_and_return_conditional_losses_67165872#
!dense_497/StatefulPartitionedCallÀ
!dense_498/StatefulPartitionedCallStatefulPartitionedCall*dense_497/StatefulPartitionedCall:output:0dense_498_6716843dense_498_6716845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_498_layer_call_and_return_conditional_losses_67166042#
!dense_498/StatefulPartitionedCall
#dropout_196/StatefulPartitionedCallStatefulPartitionedCall*dense_498/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_196_layer_call_and_return_conditional_losses_67166842%
#dropout_196/StatefulPartitionedCallÂ
!dense_499/StatefulPartitionedCallStatefulPartitionedCall,dropout_196/StatefulPartitionedCall:output:0dense_499_6716849dense_499_6716851*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_499_layer_call_and_return_conditional_losses_67166282#
!dense_499/StatefulPartitionedCall
IdentityIdentity*dense_499/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp"^dense_496/StatefulPartitionedCall"^dense_497/StatefulPartitionedCall"^dense_498/StatefulPartitionedCall"^dense_499/StatefulPartitionedCall$^dropout_196/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
: : : : : : : : 2F
!dense_496/StatefulPartitionedCall!dense_496/StatefulPartitionedCall2F
!dense_497/StatefulPartitionedCall!dense_497/StatefulPartitionedCall2F
!dense_498/StatefulPartitionedCall!dense_498/StatefulPartitionedCall2F
!dense_499/StatefulPartitionedCall!dense_499/StatefulPartitionedCall2J
#dropout_196/StatefulPartitionedCall#dropout_196/StatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

)
_user_specified_namedense_496_input"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¼
serving_default¨
K
dense_496_input8
!serving_default_dense_496_input:0ÿÿÿÿÿÿÿÿÿ
=
	dense_4990
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Áj

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
*[&call_and_return_all_conditional_losses
\_default_save_signature
]__call__"
_tf_keras_sequential
»

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__"
_tf_keras_layer
»

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"
_tf_keras_layer
»

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"
_tf_keras_layer
¥
trainable_variables
regularization_losses
 	variables
!	keras_api
*d&call_and_return_all_conditional_losses
e__call__"
_tf_keras_layer
»

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
*f&call_and_return_all_conditional_losses
g__call__"
_tf_keras_layer
I
(iter
	)decay
*learning_rate
+momentum"
	optimizer
X
0
1
2
3
4
5
"6
#7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
"6
#7"
trackable_list_wrapper
Ê

,layers
trainable_variables
-layer_regularization_losses
.metrics
/layer_metrics
regularization_losses
0non_trainable_variables
		variables
]__call__
\_default_save_signature
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
,
hserving_default"
signature_map
": 
2dense_496/kernel
:2dense_496/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

1layers
trainable_variables
2layer_regularization_losses
3metrics
4layer_metrics
regularization_losses
5non_trainable_variables
	variables
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
": 2dense_497/kernel
:2dense_497/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

6layers
trainable_variables
7layer_regularization_losses
8metrics
9layer_metrics
regularization_losses
:non_trainable_variables
	variables
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
": 2dense_498/kernel
:2dense_498/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­

;layers
trainable_variables
<layer_regularization_losses
=metrics
>layer_metrics
regularization_losses
?non_trainable_variables
	variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

@layers
trainable_variables
Alayer_regularization_losses
Bmetrics
Clayer_metrics
regularization_losses
Dnon_trainable_variables
 	variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
": 2dense_499/kernel
:2dense_499/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
­

Elayers
$trainable_variables
Flayer_regularization_losses
Gmetrics
Hlayer_metrics
%regularization_losses
Inon_trainable_variables
&	variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
N
	Mtotal
	Ncount
O	variables
P	keras_api"
_tf_keras_metric
^
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
U	keras_api"
_tf_keras_metric
^
	Vtotal
	Wcount
X
_fn_kwargs
Y	variables
Z	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
M0
N1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
-
T	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
V0
W1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
ú2÷
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716915
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716955
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716830
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716855À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÕBÒ
"__inference__wrapped_model_6716552dense_496_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
0__inference_sequential_196_layer_call_fn_6716654
0__inference_sequential_196_layer_call_fn_6716976
0__inference_sequential_196_layer_call_fn_6716997
0__inference_sequential_196_layer_call_fn_6716805À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_dense_496_layer_call_and_return_conditional_losses_6717008¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_496_layer_call_fn_6717017¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_497_layer_call_and_return_conditional_losses_6717028¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_497_layer_call_fn_6717037¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_498_layer_call_and_return_conditional_losses_6717048¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_498_layer_call_fn_6717057¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Î2Ë
H__inference_dropout_196_layer_call_and_return_conditional_losses_6717062
H__inference_dropout_196_layer_call_and_return_conditional_losses_6717074´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
-__inference_dropout_196_layer_call_fn_6717079
-__inference_dropout_196_layer_call_fn_6717084´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_dense_499_layer_call_and_return_conditional_losses_6717095¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_dense_499_layer_call_fn_6717104¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÔBÑ
%__inference_signature_wrapper_6716882dense_496_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¡
"__inference__wrapped_model_6716552{"#8¢5
.¢+
)&
dense_496_inputÿÿÿÿÿÿÿÿÿ

ª "5ª2
0
	dense_499# 
	dense_499ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_496_layer_call_and_return_conditional_losses_6717008\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_496_layer_call_fn_6717017O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_497_layer_call_and_return_conditional_losses_6717028\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_497_layer_call_fn_6717037O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_498_layer_call_and_return_conditional_losses_6717048\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_498_layer_call_fn_6717057O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_499_layer_call_and_return_conditional_losses_6717095\"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_499_layer_call_fn_6717104O"#/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
H__inference_dropout_196_layer_call_and_return_conditional_losses_6717062\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¨
H__inference_dropout_196_layer_call_and_return_conditional_losses_6717074\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dropout_196_layer_call_fn_6717079O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_dropout_196_layer_call_fn_6717084O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÂ
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716830s"#@¢=
6¢3
)&
dense_496_inputÿÿÿÿÿÿÿÿÿ

p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716855s"#@¢=
6¢3
)&
dense_496_inputÿÿÿÿÿÿÿÿÿ

p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716915j"#7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
K__inference_sequential_196_layer_call_and_return_conditional_losses_6716955j"#7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_sequential_196_layer_call_fn_6716654f"#@¢=
6¢3
)&
dense_496_inputÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_196_layer_call_fn_6716805f"#@¢=
6¢3
)&
dense_496_inputÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_196_layer_call_fn_6716976]"#7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_196_layer_call_fn_6716997]"#7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ

p

 
ª "ÿÿÿÿÿÿÿÿÿ¸
%__inference_signature_wrapper_6716882"#K¢H
¢ 
Aª>
<
dense_496_input)&
dense_496_inputÿÿÿÿÿÿÿÿÿ
"5ª2
0
	dense_499# 
	dense_499ÿÿÿÿÿÿÿÿÿ