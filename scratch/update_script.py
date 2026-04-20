import sys

file_path = 'c:/Users/Vijay/Desktop/Data-Preprocessing-and-Exploratory-Data-Analysis-Streamlit-Project/streamlit_app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []

start_idx = -1
for i, line in enumerate(lines):
    if 'st.success("✅ Training complete!")' in line:
        start_idx = i
        break

if start_idx == -1:
    print('Could not find start index')
    sys.exit(1)

new_lines = lines[:start_idx]

new_state_logic = """            # Persist full training run state
            st.session_state['trained_model'] = model
            st.session_state['trained_scaler'] = scaler
            st.session_state['trained_features'] = selected_features
            st.session_state['trained_target'] = target_col
            st.session_state['trained_task'] = task
            st.session_state['trained_model_name'] = model_name
            st.session_state['modeling_run'] = {
                'model': model, 'X': X, 'y': y,
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'y_pred': y_pred, 'y_pred_train': y_pred_train,
                'selected_features': selected_features,
                'target_col': target_col, 'task': task,
                'model_name': model_name, 'cv_folds': cv_folds
            }
        except Exception as e:
            st.error(f"Error during training: {e}")

    if 'modeling_run' in st.session_state:
        run = st.session_state['modeling_run']
        model = run['model']
        X = run['X']
        y = run['y']
        X_train = run['X_train']
        X_test = run['X_test']
        y_train = run['y_train']
        y_test = run['y_test']
        y_pred = run['y_pred']
        y_pred_train = run['y_pred_train']
        selected_features = run['selected_features']
        target_col = run['target_col']
        task = run['task']
        model_name = run['model_name']
        cv_folds = run['cv_folds']
        
        st.success("✅ Training complete!")
"""
new_lines.extend(new_state_logic.splitlines(True))

skip_idx = start_idx
while skip_idx < len(lines) and 'st.markdown("### 📊 Performance Metrics")' not in lines[skip_idx]:
    skip_idx += 1

end_idx = skip_idx
while end_idx < len(lines):
    if lines[end_idx].startswith('    return df'):
        break
    end_idx += 1

for line in lines[skip_idx:end_idx]:
    if line.startswith('            '):
        new_lines.append(line[4:])
    elif line.strip() == "":
        new_lines.append(line)
    else:
        new_lines.append(line)

new_lines.extend(lines[end_idx:])

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('File updated successfully')
