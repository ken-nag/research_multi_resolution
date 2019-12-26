total_parameters = 0
parameters_string = ""

for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
            variable_parameters *= dim.value
      total_parameters += variable_parameters
      if len(shape) == 1:
            parameters_string += ("%s %d, " % (variable.name, variable_parameters))
     else:
            parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

print(parameters_string)
print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))