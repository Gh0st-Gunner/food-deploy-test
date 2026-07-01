/**
 * "Munchin'" Figma UI - App Logic, Authentication & Onboarding
 */

document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------
    // State Management
    // ----------------------------------
    let currentUser = null; // Stores authenticated user object { username, name, targetCalories, macros, gender, age, weight, goal }
    let activeDate = null;  // Format 'YYYY-MM-DD'
    let activeCategory = 'breakfast'; // 'breakfast', 'lunch', 'dinner'
    let selectedOnboardingGender = null;
    let selectedOnboardingAge = 25;
    let selectedOnboardingWeight = 62;
    let selectedOnboardingWeightUnit = 'kg';
    let selectedOnboardingGoal = null;
    let databaseStatus = 'offline';
    let isViewingSavedMeal = false;
    
    // SVG Semi-circular Gauge constant
    const GAUGE_CIRCUMFERENCE = 283; // Circumference for r=90, half circle arc length

    // Mock meals databases by user if not exists
    const datePickerContainer = document.querySelector('.horizontal-date-slider');

    // Onboarding elements
    const welcomeContinueBtn = document.getElementById('welcome-continue-btn');
    const genderCards = document.querySelectorAll('#screen-gender .selection-card');
    const genderContinueBtn = document.getElementById('gender-continue-btn');
    const agePicker = document.getElementById('age-picker');
    const ageContinueBtn = document.getElementById('age-continue-btn');
    const weightSlider = document.getElementById('weight-slider');
    const weightValueLabel = document.getElementById('weight-value-label');
    const weightUnitBtns = document.querySelectorAll('#weight-unit-toggle .unit-btn');
    const weightContinueBtn = document.getElementById('weight-continue-btn');
    const goalCards = document.querySelectorAll('#screen-goal .selection-card');
    const goalContinueBtn = document.getElementById('goal-continue-btn');

    // Auth screen elements
    const authFlow = document.getElementById('auth-flow');
    const onboardingFlow = document.getElementById('onboarding-flow');
    const appShell = document.getElementById('app-shell');
    const signupForm = document.getElementById('signup-form');
    const signinForm = document.getElementById('signin-form');
    const authToggleLink = document.getElementById('auth-toggle-link');
    const authToggleMsg = document.getElementById('auth-toggle-msg');
    const authTitle = document.getElementById('auth-title');
    const authSubtitle = document.getElementById('auth-subtitle');

    // Header & Dashboard elements
    const statusTimeEl = document.getElementById('status-time');
    const userDisplayNameEl = document.getElementById('user-display-name');
    const profileDisplayNameEl = document.getElementById('profile-display-name');
    const profileDisplayUsernameEl = document.getElementById('profile-display-username');
    const tabNavBtns = document.querySelectorAll('.tab-nav-btn, .scan-nav-btn');
    const mealCatBtns = document.querySelectorAll('.meal-cat-btn');
    const mealCardsList = document.getElementById('meal-cards-list');

    let notifications = [];

    function pushNotification(text, type = 'info') {
        if (!currentUser) return;
        const newNotif = {
            id: 'notif-' + Date.now() + '-' + Math.floor(Math.random() * 1000),
            text,
            type,
            time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
            timestamp: Date.now(),
            read: false
        };
        notifications.push(newNotif);
        if (notifications.length > 20) {
            notifications = notifications.slice(notifications.length - 20);
        }
        localStorage.setItem(`munchin_notifications_${currentUser.username}`, JSON.stringify(notifications));
        renderNotifications();
    }

    function renderNotifications() {
        const listContainer = document.getElementById('notifications-list');
        const badge = document.getElementById('notification-badge');
        if (!listContainer) return;
        
        listContainer.innerHTML = '';
        
        if (notifications.length === 0) {
            listContainer.innerHTML = '<div class="empty-notifications">No notifications yet</div>';
            if (badge) badge.style.display = 'none';
            return;
        }
        
        const hasUnread = notifications.some(n => !n.read);
        if (badge) badge.style.display = hasUnread ? 'block' : 'none';
        
        const sorted = [...notifications].sort((a, b) => b.timestamp - a.timestamp);
        
        sorted.forEach(n => {
            const item = document.createElement('div');
            item.className = `notification-item ${n.read ? 'read' : 'unread'}`;
            
            let iconHtml = '<i class="fa-solid fa-bell"></i>';
            if (n.type === 'add') iconHtml = '<i class="fa-solid fa-circle-check" style="color:var(--accent-green)"></i>';
            else if (n.type === 'remove') iconHtml = '<i class="fa-solid fa-circle-minus" style="color:var(--accent-coral)"></i>';
            else if (n.type === 'target') iconHtml = '<i class="fa-solid fa-fire" style="color:var(--accent-yellow)"></i>';
            else if (n.type === 'recipe') iconHtml = '<i class="fa-solid fa-wand-magic-sparkles" style="color:var(--accent-coral)"></i>';
            else if (n.type === 'update') iconHtml = '<i class="fa-solid fa-circle-info" style="color:var(--accent-purple)"></i>';
            
            item.innerHTML = `
                <div class="notification-item-icon">${iconHtml}</div>
                <div class="notification-item-content">
                    <span class="notification-item-text" style="${n.read ? '' : 'font-weight: 600;'}">${n.text}</span>
                    <span class="notification-item-time">${n.time}</span>
                </div>
            `;
            listContainer.appendChild(item);
        });
    }

    function setupNotificationsUI() {
        const alertBtn = document.getElementById('header-alert-btn');
        const dropdown = document.getElementById('notification-dropdown');
        const clearBtn = document.getElementById('clear-notifications-btn');

        if (alertBtn && dropdown) {
            alertBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const isVisible = dropdown.style.display === 'block';
                if (!isVisible) {
                    dropdown.style.display = 'block';
                    notifications.forEach(n => n.read = true);
                    localStorage.setItem(`munchin_notifications_${currentUser.username}`, JSON.stringify(notifications));
                    renderNotifications();
                } else {
                    dropdown.style.display = 'none';
                }
            });

            document.addEventListener('click', (e) => {
                if (!dropdown.contains(e.target) && !alertBtn.contains(e.target)) {
                    dropdown.style.display = 'none';
                }
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                notifications = [];
                localStorage.setItem(`munchin_notifications_${currentUser.username}`, JSON.stringify(notifications));
                renderNotifications();
            });
        }
    }

    function convertSelectToCustom(selectEl) {
        if (!selectEl) return;
        
        // Hide native chevron if in select-wrapper-light
        const wrapper = selectEl.closest('.select-wrapper-light');
        if (wrapper) {
            const nativeIcon = wrapper.querySelector('.select-icon');
            if (nativeIcon) nativeIcon.style.display = 'none';
        }

        if (selectEl.dataset.customized === 'true') {
            const container = selectEl.parentElement.querySelector('.custom-select-container');
            if (container) {
                const optionsPanel = container.querySelector('.custom-select-options-panel');
                const triggerText = container.querySelector('.custom-select-trigger span');
                if (optionsPanel) {
                    optionsPanel.innerHTML = '';
                    Array.from(selectEl.options).forEach(opt => {
                        const optDiv = document.createElement('div');
                        optDiv.className = `custom-select-option ${opt.selected ? 'selected' : ''}`;
                        optDiv.textContent = opt.textContent;
                        optDiv.dataset.value = opt.value;
                        optDiv.addEventListener('click', (e) => {
                            e.stopPropagation();
                            selectEl.value = opt.value;
                            selectEl.dispatchEvent(new Event('change'));
                            triggerText.textContent = opt.textContent;
                            container.classList.remove('active');
                            container.querySelectorAll('.custom-select-option').forEach(o => o.classList.remove('selected'));
                            optDiv.classList.add('selected');
                        });
                        optionsPanel.appendChild(optDiv);
                    });
                }
                const selectedOpt = selectEl.options[selectEl.selectedIndex];
                if (selectedOpt && triggerText) {
                    triggerText.textContent = selectedOpt.textContent;
                }
            }
            return;
        }

        selectEl.style.display = 'none';
        selectEl.dataset.customized = 'true';

        const container = document.createElement('div');
        container.className = 'custom-select-container';

        const trigger = document.createElement('button');
        trigger.className = 'custom-select-trigger';
        trigger.type = 'button';
        
        const selectedOpt = selectEl.options[selectEl.selectedIndex] || selectEl.options[0];
        const triggerText = document.createElement('span');
        triggerText.textContent = selectedOpt ? selectedOpt.textContent : 'Select...';
        
        const triggerIcon = document.createElement('i');
        triggerIcon.className = 'fa-solid fa-chevron-down';
        
        trigger.appendChild(triggerText);
        trigger.appendChild(triggerIcon);

        const optionsPanel = document.createElement('div');
        optionsPanel.className = 'custom-select-options-panel';

        Array.from(selectEl.options).forEach(opt => {
            const optDiv = document.createElement('div');
            optDiv.className = `custom-select-option ${opt.selected ? 'selected' : ''}`;
            optDiv.textContent = opt.textContent;
            optDiv.dataset.value = opt.value;

            optDiv.addEventListener('click', (e) => {
                e.stopPropagation();
                selectEl.value = opt.value;
                selectEl.dispatchEvent(new Event('change'));
                triggerText.textContent = opt.textContent;
                container.classList.remove('active');
                
                container.querySelectorAll('.custom-select-option').forEach(o => o.classList.remove('selected'));
                optDiv.classList.add('selected');
            });

            optionsPanel.appendChild(optDiv);
        });

        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            document.querySelectorAll('.custom-select-container').forEach(c => {
                if (c !== container) c.classList.remove('active');
            });
            container.classList.toggle('active');
        });

        document.addEventListener('click', (e) => {
            if (!container.contains(e.target)) {
                container.classList.remove('active');
            }
        });

        container.appendChild(trigger);
        container.appendChild(optionsPanel);
        
        selectEl.parentNode.insertBefore(container, selectEl.nextSibling);
    }

    // ----------------------------------
    // 1. Initial Launch / Auth Check
    // ----------------------------------
    function init() {
        updateTime();
        setInterval(updateTime, 60000);

        // Build Age Wheel elements
        buildAgeWheel();

        // Establish Date selection
        activeDate = formatDateString(new Date());
        setupHorizontalDates();
        setupCalendarPicker();

        // Dark theme setup
        setupDarkThemeToggle();

        // Check if there is an active session
        const session = localStorage.getItem('munchin_session');
        if (session) {
            currentUser = JSON.parse(session);
            if (currentUser && currentUser.role === 'admin') {
                window.location.href = '/admin.html';
                return;
            }
            loadUserDashboard();
        } else {
            // Show Onboarding Welcome
            showFlow('onboarding');
            showOnboardingScreen('screen-welcome');
        }

        setupNotificationsUI();

        // Test database status & populate model registry
        checkBackendHealth();
        fetchAvailableModels();
        setupExploreTabs();
        setupPasswordVisibilityToggles();
        rebindAuthToggle();
    }

    function updateTime() {
        const now = new Date();
        const hrs = String(now.getHours()).padStart(2, '0');
        const mins = String(now.getMinutes()).padStart(2, '0');
        statusTimeEl.textContent = `${hrs}:${mins}`;
    }

    function setupPasswordVisibilityToggles() {
        document.querySelectorAll('.password-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const wrapper = btn.closest('.password-wrapper');
                const input = wrapper.querySelector('input');
                const icon = btn.querySelector('i');
                
                if (input.type === 'password') {
                    input.type = 'text';
                    icon.className = 'fa-regular fa-eye-slash';
                } else {
                    input.type = 'password';
                    icon.className = 'fa-regular fa-eye';
                }
            });
        });
    }

    function showFlow(flowId) {
        onboardingFlow.classList.remove('active');
        authFlow.classList.remove('active');
        appShell.classList.remove('active');

        if (flowId === 'onboarding') onboardingFlow.classList.add('active');
        else if (flowId === 'auth') authFlow.classList.add('active');
        else if (flowId === 'app') appShell.classList.add('active');
    }

    // Onboarding screens navigation
    function showOnboardingScreen(screenId) {
        const screens = document.querySelectorAll('.onboarding-screen');
        screens.forEach(s => s.classList.remove('active'));
        document.getElementById(screenId).classList.add('active');
    }

    // ----------------------------------
    // 2. Onboarding Questions Setup
    // ----------------------------------
    welcomeContinueBtn.addEventListener('click', () => {
        showOnboardingScreen('screen-gender');
    });

    // Gender card selection
    genderCards.forEach(card => {
        card.addEventListener('click', () => {
            genderCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            selectedOnboardingGender = card.dataset.gender;
            genderContinueBtn.removeAttribute('disabled');
        });
    });

    genderContinueBtn.addEventListener('click', () => {
        showOnboardingScreen('screen-age');
    });

    document.querySelectorAll('.survey-back-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const currentScreen = btn.closest('.onboarding-screen');
            if (currentScreen.id === 'screen-gender') showOnboardingScreen('screen-welcome');
            else if (currentScreen.id === 'screen-age') showOnboardingScreen('screen-gender');
            else if (currentScreen.id === 'screen-weight') showOnboardingScreen('screen-age');
            else if (currentScreen.id === 'screen-goal') showOnboardingScreen('screen-weight');
        });
    });

    // Age picker helper
    function buildAgeWheel() {
        agePicker.innerHTML = '';
        // Add empty spacer elements for look and feel
        const spacerTop = document.createElement('div');
        spacerTop.className = 'wheel-item';
        agePicker.appendChild(spacerTop);

        for (let a = 12; a <= 90; a++) {
            const item = document.createElement('div');
            item.className = 'wheel-item';
            if (a === 25) item.classList.add('selected');
            item.textContent = a;
            item.dataset.age = a;

            item.addEventListener('click', () => {
                selectedOnboardingAge = a;
                const items = agePicker.querySelectorAll('.wheel-item');
                items.forEach(i => i.classList.remove('selected'));
                item.classList.add('selected');
                
                // Scroll to center item
                item.scrollIntoView({ behavior: 'smooth', block: 'center' });
            });

            agePicker.appendChild(item);
        }

        const spacerBottom = document.createElement('div');
        spacerBottom.className = 'wheel-item';
        agePicker.appendChild(spacerBottom);
    }

    ageContinueBtn.addEventListener('click', () => {
        showOnboardingScreen('screen-weight');
    });

    // Weight slider controller
    weightSlider.addEventListener('input', (e) => {
        selectedOnboardingWeight = parseInt(e.target.value);
        weightValueLabel.textContent = selectedOnboardingWeight;
    });

    weightUnitBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            weightUnitBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedOnboardingWeightUnit = btn.dataset.unit;
            
            // Adjust slider ranges dynamically if Lbs
            if (selectedOnboardingWeightUnit === 'lbs') {
                weightSlider.min = 60;
                weightSlider.max = 330;
                // Convert kg to lbs approx
                selectedOnboardingWeight = Math.round(selectedOnboardingWeight * 2.20462);
                weightSlider.value = selectedOnboardingWeight;
            } else {
                weightSlider.min = 30;
                weightSlider.max = 150;
                // Convert lbs to kg approx
                selectedOnboardingWeight = Math.round(selectedOnboardingWeight / 2.20462);
                weightSlider.value = selectedOnboardingWeight;
            }
            weightValueLabel.textContent = selectedOnboardingWeight;
        });
    });

    weightContinueBtn.addEventListener('click', () => {
        showOnboardingScreen('screen-goal');
    });

    // Goal card selection
    goalCards.forEach(card => {
        card.addEventListener('click', () => {
            goalCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            selectedOnboardingGoal = card.dataset.goal;
            goalContinueBtn.removeAttribute('disabled');
        });
    });

    function showSignupAuthForm() {
        showFlow('auth');
        authTitle.textContent = "Create Account";
        authSubtitle.textContent = "Join us to save your BMR calorie settings";
        signupForm.style.display = 'block';
        signinForm.style.display = 'none';
        authToggleMsg.innerHTML = `Already have an account? <a href="#" id="auth-toggle-link">Sign In</a>`;
        rebindAuthToggle();
    }

    goalContinueBtn.addEventListener('click', () => {
        showSignupAuthForm();
    });

    // Handle skip buttons
    document.getElementById('gender-skip-btn').addEventListener('click', () => {
        selectedOnboardingGender = 'male';
        showOnboardingScreen('screen-age');
    });
    document.getElementById('age-skip-btn').addEventListener('click', () => {
        selectedOnboardingAge = 25;
        showOnboardingScreen('screen-weight');
    });
    document.getElementById('weight-skip-btn').addEventListener('click', () => {
        selectedOnboardingWeight = 62;
        selectedOnboardingWeightUnit = 'kg';
        showOnboardingScreen('screen-goal');
    });
    document.getElementById('goal-skip-btn').addEventListener('click', () => {
        selectedOnboardingGoal = 'maintain';
        showSignupAuthForm();
    });

    // ----------------------------------
    // 3. User Authentication Flows
    // ----------------------------------
    function rebindAuthToggle() {
        const link = document.getElementById('auth-toggle-link');
        link.addEventListener('click', (e) => {
            e.preventDefault();
            if (signupForm.style.display === 'block') {
                // Show Signin
                signupForm.style.display = 'none';
                signinForm.style.display = 'block';
                authTitle.textContent = "Welcome Back";
                authSubtitle.textContent = "Sign in to continue your calorie journey";
                authToggleMsg.innerHTML = `New to Munchin'? <a href="#" id="auth-toggle-link">Register</a>`;
            } else {
                // Show Signup
                signupForm.style.display = 'block';
                signinForm.style.display = 'none';
                authTitle.textContent = "Create Account";
                authSubtitle.textContent = "Join us to save your BMR calorie settings";
                authToggleMsg.innerHTML = `Already have an account? <a href="#" id="auth-toggle-link">Sign In</a>`;
            }
            rebindAuthToggle();
        });
    }

    // SignUp Submission
    signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('reg-name').value.trim();
        const username = document.getElementById('reg-username').value.trim().toLowerCase();
        const password = document.getElementById('reg-password').value;

        try {
            // 1. Call backend register
            const regResponse = await fetch('/api/v1/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const regData = await regResponse.json();
            if (!regResponse.ok) {
                showToast(regData.detail || 'Registration failed!', 'error');
                return;
            }

            // 2. Immediately log the new user in
            const loginResponse = await fetch('/api/v1/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const loginData = await loginResponse.json();
            if (!loginResponse.ok) {
                showToast('Registration succeeded but auto-login failed. Please sign in.', 'error');
                return;
            }

            // 3. Save BMR metrics in LocalStorage config
            let weightKg = selectedOnboardingWeight;
            if (selectedOnboardingWeightUnit === 'lbs') {
                weightKg = selectedOnboardingWeight / 2.20462;
            }
            const heightCm = (selectedOnboardingGender === 'female') ? 165 : 175;
            let bmr = 10 * weightKg + 6.25 * heightCm - 5 * selectedOnboardingAge;
            if (selectedOnboardingGender === 'female') bmr -= 161;
            else bmr += 5;
            
            let tdee = Math.round(bmr * 1.375);
            let targetCalories = tdee;
            if (selectedOnboardingGoal === 'lose') targetCalories -= 500;
            else if (selectedOnboardingGoal === 'gain') targetCalories += 400;
            targetCalories = Math.max(1200, Math.min(5000, targetCalories));

            const pTarget = Math.round((targetCalories * 0.25) / 4);
            const fTarget = Math.round((targetCalories * 0.30) / 9);
            const cTarget = Math.round((targetCalories * 0.45) / 4);

            const userConfig = {
                username,
                name: name,
                targetCalories,
                gender: selectedOnboardingGender || 'male',
                age: selectedOnboardingAge,
                weight: selectedOnboardingWeight,
                weightUnit: selectedOnboardingWeightUnit,
                goal: selectedOnboardingGoal || 'maintain',
                macros: { protein: pTarget, fats: fTarget, carbs: cTarget }
            };
            localStorage.setItem(`munchin_user_config_${username}`, JSON.stringify(userConfig));

            const sessionUser = {
                session_token: loginData.session_token,
                username: loginData.username,
                role: loginData.role,
                name: name,
                targetCalories,
                gender: userConfig.gender,
                age: userConfig.age,
                weight: userConfig.weight,
                weightUnit: userConfig.weightUnit,
                goal: userConfig.goal,
                macros: userConfig.macros
            };

            // Save active session
            localStorage.setItem('munchin_session', JSON.stringify(sessionUser));
            currentUser = sessionUser;

            // Load dashboard
            loadUserDashboard();
            showToast('Registration successful! Welcome.', 'success');

        } catch (err) {
            showToast('Unable to connect to the backend server!', 'error');
            console.error(err);
        }
    });

    // SignIn Submission
    signinForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('login-username').value.trim().toLowerCase();
        const password = document.getElementById('login-password').value;

        try {
            const response = await fetch('/api/v1/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const data = await response.json();
            if (!response.ok) {
                showToast(data.detail || 'Invalid username or password!', 'error');
                return;
            }

            // Check role: if admin, redirect to admin.html
            if (data.role === 'admin') {
                localStorage.setItem('admin_token', data.session_token);
                localStorage.setItem('admin_username', data.username);
                localStorage.setItem('admin_role', data.role);
                showToast('Welcome Admin! Redirecting to Dashboard...', 'success');
                setTimeout(() => {
                    window.location.href = '/admin.html';
                }, 1000);
                return;
            }

            // Normal user flow: fetch target calories from localStorage or assign default BMR
            const configKey = `munchin_user_config_${username}`;
            let userConfig = JSON.parse(localStorage.getItem(configKey));
            
            if (!userConfig) {
                const targetCalories = 2200;
                const pTarget = Math.round((targetCalories * 0.25) / 4);
                const fTarget = Math.round((targetCalories * 0.30) / 9);
                const cTarget = Math.round((targetCalories * 0.45) / 4);
                userConfig = {
                    username,
                    name: username,
                    targetCalories,
                    gender: 'male',
                    age: 25,
                    weight: 62,
                    weightUnit: 'kg',
                    goal: 'maintain',
                    macros: { protein: pTarget, fats: fTarget, carbs: cTarget }
                };
                localStorage.setItem(configKey, JSON.stringify(userConfig));
            }

            const sessionUser = {
                session_token: data.session_token,
                username: data.username,
                role: data.role,
                name: userConfig.name || data.username,
                targetCalories: userConfig.targetCalories,
                gender: userConfig.gender,
                age: userConfig.age,
                weight: userConfig.weight,
                weightUnit: userConfig.weightUnit,
                goal: userConfig.goal,
                macros: userConfig.macros
            };

            // Save active session
            localStorage.setItem('munchin_session', JSON.stringify(sessionUser));
            currentUser = sessionUser;

            loadUserDashboard();
            showToast(`Welcome back, ${currentUser.name}!`, 'success');

        } catch (err) {
            showToast('Unable to connect to the backend server!', 'error');
            console.error(err);
        }
    });

    // SignOut Event
    document.getElementById('logout-btn').addEventListener('click', () => {
        localStorage.removeItem('munchin_session');
        currentUser = null;
        
        // Reset flows
        showFlow('onboarding');
        showOnboardingScreen('screen-welcome');
        
        // Reset forms inputs
        signupForm.reset();
        signinForm.reset();
        
        showToast('You have logged out.', 'success');
    });

    // Recalculate BMR option in menu
    document.getElementById('menu-survey').addEventListener('click', () => {
        showFlow('onboarding');
        showOnboardingScreen('screen-gender');
    });

    // Adjust Target Calories Modal
    const customTargetCaloriesInput = document.getElementById('custom-target-calories-val');
    if (customTargetCaloriesInput) {
        customTargetCaloriesInput.addEventListener('focus', () => {
            customTargetCaloriesInput.select();
        });
    }

    document.getElementById('menu-intake-goal').addEventListener('click', () => {
        if (customTargetCaloriesInput) {
            customTargetCaloriesInput.value = currentUser.targetCalories;
        }
        document.getElementById('custom-target-modal').classList.add('active');
    });

    document.getElementById('close-target-modal').addEventListener('click', () => {
        document.getElementById('custom-target-modal').classList.remove('active');
    });
    document.getElementById('cancel-target-modal-btn').addEventListener('click', () => {
        document.getElementById('custom-target-modal').classList.remove('active');
    });

    document.getElementById('save-target-modal-btn').addEventListener('click', () => {
        const val = parseInt(document.getElementById('custom-target-calories-val').value);
        if (val >= 800 && val <= 8000) {
            currentUser.targetCalories = val;
            
            // Adjust macros goals proportionally
            currentUser.macros.protein = Math.round((val * 0.25) / 4);
            currentUser.macros.fats = Math.round((val * 0.30) / 9);
            currentUser.macros.carbs = Math.round((val * 0.45) / 4);

            // Update user config in localStorage
            const configKey = `munchin_user_config_${currentUser.username}`;
            const userConfig = JSON.parse(localStorage.getItem(configKey)) || {};
            userConfig.targetCalories = val;
            userConfig.macros = currentUser.macros;
            localStorage.setItem(configKey, JSON.stringify(userConfig));
            // Update session cache
            localStorage.setItem('munchin_session', JSON.stringify(currentUser));

            document.getElementById('custom-target-modal').classList.remove('active');
            
            // Re-render Dashboard / Profiles
            loadUserDashboard();
            showToast('Calorie goals adjusted successfully!', 'success');
            pushNotification(`Calorie target updated to ${val} kcal.`, 'target');
        } else {
            showToast('Goal must be between 800 and 8000 kcal.', 'error');
        }
    });

    // Error Modal setup
    const errModalEl = document.getElementById('error-modal');
    const closeErrModal = () => {
        if (errModalEl) errModalEl.classList.remove('active');
    };
    if (document.getElementById('close-error-modal')) {
        document.getElementById('close-error-modal').addEventListener('click', closeErrModal);
    }
    if (document.getElementById('close-error-modal-btn')) {
        document.getElementById('close-error-modal-btn').addEventListener('click', closeErrModal);
    }
    if (document.getElementById('copy-error-btn')) {
        document.getElementById('copy-error-btn').addEventListener('click', () => {
            const tracebackArea = document.getElementById('error-traceback');
            if (tracebackArea) {
                tracebackArea.select();
                tracebackArea.setSelectionRange(0, 99999);
                navigator.clipboard.writeText(tracebackArea.value)
                    .then(() => {
                        showToast('Diagnostic details copied to clipboard!', 'success');
                    })
                    .catch(() => {
                        try {
                            document.execCommand('copy');
                            showToast('Diagnostic details copied to clipboard!', 'success');
                        } catch (e) {
                            showToast('Failed to copy. Copy manually.', 'error');
                        }
                    });
            }
        });
    }

    function showErrorModal(errorDetails) {
        const tracebackArea = document.getElementById('error-traceback');
        if (tracebackArea) {
            tracebackArea.value = errorDetails || 'Unknown error occurred during analysis.';
        }
        if (errModalEl) {
            errModalEl.classList.add('active');
        }
    }

    // Simple alerts menu
    document.getElementById('menu-favorites').addEventListener('click', () => {
        showToast('No favorite foods saved yet!', 'success');
    });

    // ----------------------------------
    // 4. Seeding user logs
    // ----------------------------------
    function seedUserDemoData(username) {
        const todayStr = getFormattedDate(0);
        const yesterdayStr = getFormattedDate(-1);

        // Pre-fill Yesterday
        const yMeals = [
            {
                id: 'meal-d1',
                name: 'Phở Bò (Beef Pho)',
                calories: 680,
                protein: 32,
                carbs: 85,
                fat: 16,
                time: '08:15',
                category: 'breakfast'
            },
            {
                id: 'meal-d2',
                name: 'Cơm Tấm Sườn',
                calories: 820,
                protein: 36,
                carbs: 95,
                fat: 26,
                time: '12:45',
                category: 'lunch'
            }
        ];
        localStorage.setItem(`munchin_meals_${username}_${yesterdayStr}`, JSON.stringify(yMeals));

        // Pre-fill Today
        const tMeals = [
            {
                id: 'meal-d3',
                name: 'Bánh Mì Kẹp Thịt',
                calories: 490,
                protein: 18,
                carbs: 62,
                fat: 15,
                time: '07:30',
                category: 'breakfast'
            }
        ];
        localStorage.setItem(`munchin_meals_${username}_${todayStr}`, JSON.stringify(tMeals));
    }

    // ----------------------------------
    // 5. Dashboard Load & Render
    // ----------------------------------
    function loadUserDashboard() {
        showFlow('app');

        // Populate display labels
        userDisplayNameEl.textContent = currentUser.name;
        profileDisplayNameEl.textContent = currentUser.name;
        profileDisplayUsernameEl.textContent = `@${currentUser.username}`;
        document.getElementById('profile-avatar').src = `https://api.dicebear.com/7.x/adventurer/svg?seed=${currentUser.username}`;
        document.getElementById('header-avatar').src = `https://api.dicebear.com/7.x/adventurer/svg?seed=${currentUser.username}`;

        // Profile details
        document.getElementById('prof-stat-cal').textContent = `${currentUser.targetCalories} kcal`;
        document.getElementById('prof-stat-goal').textContent = currentUser.goal;
        document.getElementById('prof-stat-weight').textContent = `${currentUser.weight} ${currentUser.weightUnit === 'lbs' ? 'Lbs' : 'Kg'}`;

        // Initialize active screen back to Home
        showAppScreen('home');

        // Load notifications from local storage
        notifications = JSON.parse(localStorage.getItem(`munchin_notifications_${currentUser.username}`)) || [];
        renderNotifications();

        // Setup Weight & Exercise Log bindings
        setupWeightAndExerciseLogs();

        // Render dashboard values
        updateDashboardValues();
    }

    function updateDashboardValues() {
        if (!currentUser) return;
        const mealsKey = `munchin_meals_${currentUser.username}_${activeDate}`;
        const meals = JSON.parse(localStorage.getItem(mealsKey)) || [];

        // Sum consumed stats
        const totalCal = meals.reduce((s, m) => s + m.calories, 0);
        const totalP = meals.reduce((s, m) => s + m.protein, 0);
        const totalC = meals.reduce((s, m) => s + m.carbs, 0);
        const totalF = meals.reduce((s, m) => s + m.fat, 0);

        // Sum workouts (burned calories)
        const workoutsKey = `munchin_workouts_${currentUser.username}_${activeDate}`;
        const workouts = JSON.parse(localStorage.getItem(workoutsKey)) || [];
        const totalBurned = workouts.reduce((s, w) => s + w.calories, 0);

        // Update semi-circular gauge SVG
        document.getElementById('gauge-cal-completed').textContent = totalCal;
        document.getElementById('gauge-cal-target').textContent = currentUser.targetCalories;

        const fillPercent = Math.min(1.0, totalCal / currentUser.targetCalories);
        // Dashoffset: 0 is full, 283 is empty
        const dashOffset = GAUGE_CIRCUMFERENCE * (1 - fillPercent);
        document.getElementById('gauge-fill-arc').style.strokeDashoffset = dashOffset;

        // Update Gauge Stats Row (Budget, Left, Exercise)
        const budget = currentUser.targetCalories;
        const left = budget - totalCal + totalBurned;

        const budgetValEl = document.getElementById('gauge-budget-val');
        const leftValEl = document.getElementById('gauge-left-val');
        const exerciseValEl = document.getElementById('gauge-exercise-val');
        if (budgetValEl) budgetValEl.textContent = budget;
        if (leftValEl) leftValEl.textContent = left;
        if (exerciseValEl) exerciseValEl.textContent = totalBurned;

        // Update Macros
        document.getElementById('macro-p-val').textContent = totalP;
        document.getElementById('macro-p-tgt').textContent = currentUser.macros.protein;
        const pPercent = Math.min(100, (totalP / currentUser.macros.protein) * 100);
        document.getElementById('macro-bar-p').style.width = `${pPercent}%`;

        document.getElementById('macro-f-val').textContent = totalF;
        document.getElementById('macro-f-tgt').textContent = currentUser.macros.fats;
        const fPercent = Math.min(100, (totalF / currentUser.macros.fats) * 100);
        document.getElementById('macro-bar-f').style.width = `${fPercent}%`;

        document.getElementById('macro-c-val').textContent = totalC;
        document.getElementById('macro-c-tgt').textContent = currentUser.macros.carbs;
        const cPercent = Math.min(100, (totalC / currentUser.macros.carbs) * 100);
        document.getElementById('macro-bar-c').style.width = `${cPercent}%`;

        // Render workouts & weight logs
        renderDashboardWorkouts();
        renderDashboardWeight();

        // Render active meals cards
        renderMealCategoryCards();
    }

    /* ----------------------------------
     * Pitch-Black Dark Mode & Logs Wiring
     * ---------------------------------- */
    function setupDarkThemeToggle() {
        const toggle = document.getElementById('dark-theme-toggle');
        const savedTheme = localStorage.getItem('munchin_theme_black');

        if (savedTheme === 'true') {
            document.body.classList.add('theme-black');
            if (toggle) toggle.checked = true;
        } else {
            document.body.classList.remove('theme-black');
            if (toggle) toggle.checked = false;
        }

        if (toggle) {
            toggle.addEventListener('change', () => {
                if (toggle.checked) {
                    document.body.classList.add('theme-black');
                    localStorage.setItem('munchin_theme_black', 'true');
                    showToast('Pitch-Black dark mode enabled!', 'success');
                } else {
                    document.body.classList.remove('theme-black');
                    localStorage.setItem('munchin_theme_black', 'false');
                    showToast('Light mode enabled!', 'success');
                }
            });
        }
    }

    function setupWeightAndExerciseLogs() {
        const weightModal = document.getElementById('weight-log-modal');
        const exerciseModal = document.getElementById('exercise-log-modal');

        const openWeightBtn = document.getElementById('btn-open-weight-log');
        const closeWeightBtn = document.getElementById('close-weight-modal');
        const cancelWeightBtn = document.getElementById('cancel-weight-modal-btn');
        const saveWeightBtn = document.getElementById('save-weight-modal-btn');
        const weightInput = document.getElementById('weight-log-input');

        const openExerciseBtn = document.getElementById('btn-open-exercise-log');
        const closeExerciseBtn = document.getElementById('close-exercise-modal');
        const cancelExerciseBtn = document.getElementById('cancel-exercise-modal-btn');
        const saveExerciseBtn = document.getElementById('save-exercise-modal-btn');

        const activitySelect = document.getElementById('exercise-activity-select');
        const customNameGroup = document.getElementById('custom-exercise-name-group');
        const customNameInput = document.getElementById('custom-exercise-name');
        const customCalGroup = document.getElementById('custom-exercise-calories-group');
        const customCalInput = document.getElementById('custom-exercise-calories');
        const durationInput = document.getElementById('exercise-duration-input');
        const estBurnSpan = document.getElementById('exercise-estimated-burn');

        // Update units label inside weight modal
        if (currentUser) {
            const unitLabels = document.querySelectorAll('.weight-unit-label');
            unitLabels.forEach(el => el.textContent = currentUser.weightUnit || 'kg');
        }

        // Weight log buttons
        if (openWeightBtn && weightModal) {
            // Remove previous event listener clones by replacing the element or standard binding
            const newOpenWeightBtn = openWeightBtn.cloneNode(true);
            openWeightBtn.parentNode.replaceChild(newOpenWeightBtn, openWeightBtn);
            newOpenWeightBtn.addEventListener('click', () => {
                weightInput.value = currentUser ? currentUser.weight : '';
                const unitLabels = document.querySelectorAll('.weight-unit-label');
                unitLabels.forEach(el => el.textContent = currentUser ? currentUser.weightUnit : 'kg');
                renderDashboardWeight();
                weightModal.classList.add('active');
            });
        }
        const closeWeight = () => {
            if (weightModal) weightModal.classList.remove('active');
        };
        if (closeWeightBtn) {
            const newCloseWeightBtn = closeWeightBtn.cloneNode(true);
            closeWeightBtn.parentNode.replaceChild(newCloseWeightBtn, closeWeightBtn);
            newCloseWeightBtn.addEventListener('click', closeWeight);
        }
        if (cancelWeightBtn) {
            const newCancelWeightBtn = cancelWeightBtn.cloneNode(true);
            cancelWeightBtn.parentNode.replaceChild(newCancelWeightBtn, cancelWeightBtn);
            newCancelWeightBtn.addEventListener('click', closeWeight);
        }

        if (saveWeightBtn) {
            const newSaveWeightBtn = saveWeightBtn.cloneNode(true);
            saveWeightBtn.parentNode.replaceChild(newSaveWeightBtn, saveWeightBtn);
            newSaveWeightBtn.addEventListener('click', () => {
                const val = parseFloat(weightInput.value);
                if (isNaN(val) || val <= 0) {
                    showToast('Please enter a valid weight!', 'error');
                    return;
                }
                addWeightLog(val);
                closeWeight();
            });
        }

        // Exercise log buttons
        if (openExerciseBtn && exerciseModal) {
            const newOpenExerciseBtn = openExerciseBtn.cloneNode(true);
            openExerciseBtn.parentNode.replaceChild(newOpenExerciseBtn, openExerciseBtn);
            newOpenExerciseBtn.addEventListener('click', () => {
                activitySelect.value = 'Walking';
                durationInput.value = 30;
                customNameInput.value = '';
                customCalInput.value = '';
                if (customNameGroup) customNameGroup.style.display = 'none';
                if (customCalGroup) customCalGroup.style.display = 'none';
                updateBurnEstimation();
                exerciseModal.classList.add('active');
            });
        }
        const closeExercise = () => {
            if (exerciseModal) exerciseModal.classList.remove('active');
        };
        if (closeExerciseBtn) {
            const newCloseExerciseBtn = closeExerciseBtn.cloneNode(true);
            closeExerciseBtn.parentNode.replaceChild(newCloseExerciseBtn, closeExerciseBtn);
            newCloseExerciseBtn.addEventListener('click', closeExercise);
        }
        if (cancelExerciseBtn) {
            const newCancelExerciseBtn = cancelExerciseBtn.cloneNode(true);
            cancelExerciseBtn.parentNode.replaceChild(newCancelExerciseBtn, cancelExerciseBtn);
            newCancelExerciseBtn.addEventListener('click', closeExercise);
        }

        // Helper to calculate burned calories estimate
        function updateBurnEstimation() {
            const selectedOpt = activitySelect.options[activitySelect.selectedIndex];
            const rateStr = selectedOpt.dataset.rate;
            const duration = parseFloat(durationInput.value) || 0;

            if (rateStr === 'custom') {
                if (customNameGroup) customNameGroup.style.display = 'block';
                if (customCalGroup) customCalGroup.style.display = 'block';
                const customCal = parseFloat(customCalInput.value) || 0;
                estBurnSpan.textContent = customCal;
            } else {
                if (customNameGroup) customNameGroup.style.display = 'none';
                if (customCalGroup) customCalGroup.style.display = 'none';
                const rate = parseFloat(rateStr) || 0;
                estBurnSpan.textContent = Math.round(duration * rate);
            }
        }

        if (activitySelect) {
            activitySelect.addEventListener('change', updateBurnEstimation);
        }
        if (durationInput) {
            durationInput.addEventListener('input', updateBurnEstimation);
            durationInput.addEventListener('focus', () => durationInput.select());
        }
        if (customCalInput) {
            customCalInput.addEventListener('input', updateBurnEstimation);
            customCalInput.addEventListener('focus', () => customCalInput.select());
        }
        if (weightInput) {
            weightInput.addEventListener('focus', () => weightInput.select());
        }

        if (saveExerciseBtn) {
            const newSaveExerciseBtn = saveExerciseBtn.cloneNode(true);
            saveExerciseBtn.parentNode.replaceChild(newSaveExerciseBtn, saveExerciseBtn);
            newSaveExerciseBtn.addEventListener('click', () => {
                const selectedOpt = activitySelect.options[activitySelect.selectedIndex];
                const rateStr = selectedOpt.dataset.rate;
                const duration = parseFloat(durationInput.value) || 0;

                if (isNaN(duration) || duration <= 0) {
                    showToast('Please enter a valid duration!', 'error');
                    return;
                }

                let activityName = selectedOpt.value;
                let calories = 0;

                if (rateStr === 'custom') {
                    activityName = customNameInput.value.trim() || 'Custom Activity';
                    calories = parseFloat(customCalInput.value) || 0;
                    if (calories <= 0) {
                        showToast('Please enter burned calories!', 'error');
                        return;
                    }
                } else {
                    const rate = parseFloat(rateStr) || 0;
                    calories = Math.round(duration * rate);
                }

                addWorkoutLog(activityName, duration, calories);
                closeExercise();
            });
        }
    }

    function addWeightLog(weightValue) {
        if (!currentUser) return;
        const weightKey = `munchin_weight_${currentUser.username}`;
        const weightHistory = JSON.parse(localStorage.getItem(weightKey)) || [];

        // Remove previous log of the same day if any, and add new
        const index = weightHistory.findIndex(w => w.date === activeDate);
        const logItem = { date: activeDate, weight: parseFloat(weightValue) };
        if (index !== -1) {
            weightHistory[index] = logItem;
        } else {
            weightHistory.push(logItem);
        }
        localStorage.setItem(weightKey, JSON.stringify(weightHistory));

        // Update current weight in user object
        currentUser.weight = parseFloat(weightValue);

        // Sync with user config in localStorage
        const configKey = `munchin_user_config_${currentUser.username}`;
        const userConfig = JSON.parse(localStorage.getItem(configKey)) || {};
        userConfig.weight = currentUser.weight;
        localStorage.setItem(configKey, JSON.stringify(userConfig));
        localStorage.setItem('munchin_session', JSON.stringify(currentUser));

        // Update display on profile page
        document.getElementById('prof-stat-weight').textContent = `${currentUser.weight} ${currentUser.weightUnit === 'lbs' ? 'Lbs' : 'Kg'}`;

        updateDashboardValues();
        showToast(`Weight of ${weightValue} ${currentUser.weightUnit === 'lbs' ? 'Lbs' : 'Kg'} logged for today!`, 'success');
        pushNotification(`Logged weight: ${weightValue} ${currentUser.weightUnit === 'lbs' ? 'Lbs' : 'Kg'}`, 'update');
    }

    function renderDashboardWeight() {
        const weightKey = `munchin_weight_${currentUser.username}`;
        const weightHistory = JSON.parse(localStorage.getItem(weightKey)) || [];
        const currentWeightValEl = document.getElementById('dashboard-current-weight-val');
        if (!currentWeightValEl) return;

        // Find weight for today specifically
        const todayWeightLog = weightHistory.find(w => w.date === activeDate);
        if (todayWeightLog) {
            currentWeightValEl.textContent = `${todayWeightLog.weight} ${currentUser.weightUnit === 'lbs' ? 'Lbs' : 'Kg'}`;
        } else {
            // Fallback to latest weight in history, or user profile weight
            if (weightHistory.length > 0) {
                const sorted = [...weightHistory].sort((a, b) => new Date(b.date) - new Date(a.date));
                currentWeightValEl.textContent = `${sorted[0].weight} ${currentUser.weightUnit === 'lbs' ? 'Lbs' : 'Kg'} (last)`;
            } else {
                currentWeightValEl.textContent = `${currentUser.weight} ${currentUser.weightUnit === 'lbs' ? 'Lbs' : 'Kg'}`;
            }
        }

        // Render weight history list inside the modal
        const historyListContainer = document.getElementById('weight-history-list');
        if (historyListContainer) {
            historyListContainer.innerHTML = '';
            if (weightHistory.length === 0) {
                historyListContainer.innerHTML = '<span style="color:var(--text-light-gray); font-style:italic; font-size:11px;">No weight history logged yet.</span>';
            } else {
                const sorted = [...weightHistory].sort((a, b) => new Date(b.date) - new Date(a.date));
                sorted.forEach(w => {
                    const item = document.createElement('div');
                    item.style.display = 'flex';
                    item.style.justifyContent = 'space-between';
                    item.style.alignItems = 'center';
                    item.style.backgroundColor = 'var(--bg-light)';
                    item.style.padding = '6px 10px';
                    item.style.borderRadius = '8px';
                    item.style.fontSize = '11px';

                    item.innerHTML = `
                        <div>
                            <strong>${w.date}</strong>
                        </div>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span style="font-weight:600; color:var(--accent-purple);">${w.weight} ${currentUser.weightUnit === 'lbs' ? 'Lbs' : 'Kg'}</span>
                            <button class="delete-weight-btn" data-date="${w.date}" style="background:none; border:none; color:#F05C3B; cursor:pointer;" title="Delete log"><i class="fa-solid fa-trash-can"></i></button>
                        </div>
                    `;

                    item.querySelector('.delete-weight-btn').addEventListener('click', (e) => {
                        e.stopPropagation();
                        document.getElementById('delete-weight-date').value = w.date;
                        document.getElementById('delete-weight-modal').classList.add('active');
                    });

                    historyListContainer.appendChild(item);
                });
            }
        }
    }

    function removeWeightLog(date) {
        const weightKey = `munchin_weight_${currentUser.username}`;
        let weightHistory = JSON.parse(localStorage.getItem(weightKey)) || [];
        weightHistory = weightHistory.filter(w => w.date !== date);
        localStorage.setItem(weightKey, JSON.stringify(weightHistory));

        renderDashboardWeight();
        updateDashboardValues();
        showToast('Weight log deleted.', 'success');
    }

    function addWorkoutLog(name, duration, calories) {
        if (!currentUser) return;
        const workoutsKey = `munchin_workouts_${currentUser.username}_${activeDate}`;
        const workouts = JSON.parse(localStorage.getItem(workoutsKey)) || [];

        const newWorkout = {
            id: 'workout-' + Date.now(),
            name,
            duration: parseInt(duration),
            calories: parseInt(calories)
        };
        workouts.push(newWorkout);
        localStorage.setItem(workoutsKey, JSON.stringify(workouts));

        updateDashboardValues();
        showToast(`Logged workout "${name}" for today!`, 'success');
        pushNotification(`Logged exercise: ${name} (${calories} kcal)`, 'add');
    }

    function renderDashboardWorkouts() {
        const workoutsKey = `munchin_workouts_${currentUser.username}_${activeDate}`;
        const workouts = JSON.parse(localStorage.getItem(workoutsKey)) || [];
        const container = document.getElementById('dashboard-today-workouts-list');
        if (!container) return;

        container.innerHTML = '';
        if (workouts.length === 0) {
            container.innerHTML = '<span style="color: var(--text-light-gray); font-style: italic; font-size: 11px;">No exercise logged today.</span>';
            return;
        }

        workouts.forEach(w => {
            const item = document.createElement('div');
            item.style.display = 'flex';
            item.style.justifyContent = 'space-between';
            item.style.alignItems = 'center';
            item.style.backgroundColor = 'var(--bg-light)';
            item.style.padding = '6px 10px';
            item.style.borderRadius = '8px';
            item.style.fontSize = '11px';

            item.innerHTML = `
                <div style="display:flex; align-items:center; gap:6px;">
                    <i class="fa-solid fa-person-running" style="color:var(--accent-green)"></i>
                    <strong>${w.name}</strong>
                    <span style="color:var(--text-muted)">(${w.duration} min)</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-weight:600; color:var(--accent-green);">${w.calories} kcal</span>
                    <button class="delete-workout-btn" data-id="${w.id}" style="background:none; border:none; color:#F05C3B; cursor:pointer;" title="Delete workout"><i class="fa-solid fa-trash-can"></i></button>
                </div>
            `;

            item.querySelector('.delete-workout-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                document.getElementById('delete-workout-id').value = w.id;
                document.getElementById('delete-workout-modal').classList.add('active');
            });

            container.appendChild(item);
        });
    }

    function removeWorkout(workoutId) {
        const workoutsKey = `munchin_workouts_${currentUser.username}_${activeDate}`;
        let workouts = JSON.parse(localStorage.getItem(workoutsKey)) || [];
        const workoutItem = workouts.find(w => w.id === workoutId);
        const name = workoutItem ? workoutItem.name : 'Exercise';
        workouts = workouts.filter(w => w.id !== workoutId);
        localStorage.setItem(workoutsKey, JSON.stringify(workouts));

        updateDashboardValues();
        showToast('Workout deleted successfully.', 'success');
        pushNotification(`Removed exercise "${name}" from your journey.`, 'remove');
    }

    function renderMealCategoryCards() {
        const mealsKey = `munchin_meals_${currentUser.username}_${activeDate}`;
        const meals = JSON.parse(localStorage.getItem(mealsKey)) || [];
        const categoryMeals = meals.filter(m => m.category === activeCategory);

        // Clear existing cards
        const emptyState = mealCardsList.querySelector('.empty-state-card');
        mealCardsList.querySelectorAll('.meal-card-item').forEach(el => el.remove());

        if (categoryMeals.length === 0) {
            emptyState.style.display = 'flex';
            return;
        }

        emptyState.style.display = 'none';

        categoryMeals.forEach(meal => {
            const card = document.createElement('div');
            card.className = 'meal-card-item';
            card.dataset.id = meal.id;

            let emoji = '🍜';
            if (meal.name.toLowerCase().includes('bánh mì') || meal.name.toLowerCase().includes('banh mi')) emoji = '🥖';
            else if (meal.name.toLowerCase().includes('cơm') || meal.name.toLowerCase().includes('rice')) emoji = '🍛';
            else if (meal.name.toLowerCase().includes('salad') || meal.name.toLowerCase().includes('gỏi')) emoji = '🥗';
            else if (meal.name.toLowerCase().includes('trứng') || meal.name.toLowerCase().includes('egg')) emoji = '🍳';
            else if (meal.name.toLowerCase().includes('cà phê') || meal.name.toLowerCase().includes('coffee')) emoji = '☕';

            card.innerHTML = `
                <div class="meal-card-item-left">
                    <div class="meal-card-icon">${emoji}</div>
                    <div class="meal-card-details">
                        <span class="meal-card-title">${meal.name}</span>
                        <span class="meal-card-subtitle"><i class="fa-regular fa-clock"></i> ${meal.time} | P: ${meal.protein}g C: ${meal.carbs}g F: ${meal.fat}g</span>
                    </div>
                </div>
                <div class="meal-card-item-right">
                    <span class="meal-card-calories">${meal.calories} kcal</span>
                    <button class="delete-meal-btn" style="color:#F05C3B" title="Delete meal"><i class="fa-solid fa-trash-can"></i></button>
                </div>
            `;

            // Delete meal action
            card.querySelector('.delete-meal-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                document.getElementById('delete-meal-id').value = meal.id;
                document.getElementById('delete-meal-modal').classList.add('active');
            });

            // View meal details action on click
            card.style.cursor = 'pointer';
            card.addEventListener('click', () => {
                isViewingSavedMeal = true;
                
                const result = {
                    name: meal.name,
                    confidence: meal.confidence || 100,
                    portion: meal.portion || '1.0 portion',
                    calories: meal.calories,
                    protein: meal.protein,
                    carbs: meal.carbs,
                    fat: meal.fat !== undefined ? meal.fat : (meal.fats || 0),
                    ingredients: meal.ingredients || [],
                    overlay_url: meal.overlay_url || null,
                    depth_url: meal.depth_url || null,
                    image_src: meal.image_src || 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&q=80&w=400',
                    isMock: !meal.depth_url && (!meal.ingredients || meal.ingredients.length === 0)
                };
                
                showAppScreen('scan');
                
                const isAccurate = !!(meal.ingredients && meal.ingredients.length > 0) || !!meal.depth_url;
                displayScanResult(result, isAccurate);
            });

            mealCardsList.appendChild(card);
        });
    }

    function removeMeal(mealId) {
        const mealsKey = `munchin_meals_${currentUser.username}_${activeDate}`;
        let meals = JSON.parse(localStorage.getItem(mealsKey)) || [];
        const mealItem = meals.find(m => m.id === mealId);
        const mealName = mealItem ? mealItem.name : 'Meal';
        meals = meals.filter(m => m.id !== mealId);
        localStorage.setItem(mealsKey, JSON.stringify(meals));
        
        updateDashboardValues();
        showToast('Meal item deleted successfully.', 'success');
        pushNotification(`Removed meal "${mealName}" from your journey.`, 'remove');
    }

    function addMealItem(name, cal, p, c, f, category, imageSrc = null, portion = null, ingredients = null, depthUrl = null) {
        if (!currentUser) return;
        const mealsKey = `munchin_meals_${currentUser.username}_${activeDate}`;
        const meals = JSON.parse(localStorage.getItem(mealsKey)) || [];

        const now = new Date();
        const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

        const newItem = {
            id: 'meal-' + Date.now(),
            name,
            calories: parseInt(cal) || 0,
            protein: parseInt(p) || 0,
            carbs: parseInt(c) || 0,
            fat: parseInt(f) || 0,
            time: timeStr,
            category: category || activeCategory,
            image_src: imageSrc,
            portion: portion,
            ingredients: ingredients,
            depth_url: depthUrl
        };

        meals.push(newItem);
        localStorage.setItem(mealsKey, JSON.stringify(meals));

        updateDashboardValues();
        showToast(`Logged "${name}" (${cal} kcal) to ${category || activeCategory}`, 'success');
        pushNotification(`Logged "${name}" (${cal} kcal) to ${category || activeCategory}`, 'add');

        // Close scan results panel if it was open
        const resultPanel = document.getElementById('result-panel-revamp');
        if (resultPanel) {
            resultPanel.style.display = 'none';
        }

        // Show custom success modal
        const successModal = document.getElementById('success-modal');
        if (successModal) {
            const catNames = { 'breakfast': 'Bữa Sáng', 'lunch': 'Bữa Trưa', 'dinner': 'Bữa Tối' };
            const targetCat = category || activeCategory;
            const viCat = catNames[targetCat] || targetCat;
            const successMsg = `Món ăn "${name}" (${cal} kcal) đã được lưu thành công vào ${viCat}!`;
            
            const msgEl = document.getElementById('success-modal-msg');
            if (msgEl) msgEl.textContent = successMsg;
            
            successModal.classList.add('active');
        } else {
            resetScanDropzone();
            showAppScreen('home');
        }
    }

    // ----------------------------------
    // 6. Navigation Control & Tab routing
    // ----------------------------------
    function showAppScreen(screenId) {
        const screens = document.querySelectorAll('.app-screen');
        screens.forEach(s => s.classList.remove('active'));
        document.getElementById(`screen-${screenId}`).classList.add('active');

        // Highlight active bottom nav btn
        tabNavBtns.forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.screen === screenId) btn.classList.add('active');
        });

        // Toggle sticky header visibility based on screen ID (only show on Home)
        const header = document.getElementById('main-header');
        if (screenId === 'home') {
            header.style.display = 'flex';
        } else {
            header.style.display = 'none';
        }

        // Draw weekly graph if Reports screen is selected
        if (screenId === 'reports') {
            setTimeout(drawProgressChart, 100);
        }
    }

    tabNavBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const screen = btn.dataset.screen;
            if (screen === 'scan') {
                isViewingSavedMeal = false;
                resetScanDropzone();
            }
            showAppScreen(screen);
        });
    });

    // Meal categories toggle breakfast / lunch / dinner
    mealCatBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            mealCatBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeCategory = btn.dataset.category;
            renderMealCategoryCards();
        });
    });

    // ----------------------------------
    // 7. Date picker generator
    // ----------------------------------
    function setupHorizontalDates(centerDate = new Date()) {
        datePickerContainer.innerHTML = '';
        const todayKey = formatDateString(new Date());
        
        for (let offset = -3; offset <= 3; offset++) {
            const dateObj = new Date(centerDate);
            dateObj.setDate(dateObj.getDate() + offset);

            const dayName = dateObj.toLocaleDateString('en-US', { weekday: 'short' });
            const dayNum = dateObj.getDate();
            const dateKey = formatDateString(dateObj);

            const item = document.createElement('div');
            item.className = 'date-slide-item';
            
            if (dateKey === todayKey) {
                item.classList.add('is-today');
            }
            if (dateKey === activeDate) {
                item.classList.add('active');
            }

            item.innerHTML = `
                <span class="day">${dayName}</span>
                <span class="num">${dayNum}</span>
            `;

            item.addEventListener('click', () => {
                activeDate = dateKey;
                setupHorizontalDates(dateObj);
                updateDashboardValues();
            });

            datePickerContainer.appendChild(item);
        }
    }

    function getFormattedDate(dayOffset) {
        const d = new Date();
        d.setDate(d.getDate() + dayOffset);
        return formatDateString(d);
    }

    function formatDateString(date) {
        const y = date.getFullYear();
        const m = String(date.getMonth() + 1).padStart(2, '0');
        const d = String(date.getDate()).padStart(2, '0');
        return `${y}-${m}-${d}`;
    }

    // ----------------------------------
    // 8. Canvas line chart drawings (Reports)
    // ----------------------------------
    function drawProgressChart() {
        const canvas = document.getElementById('progress-line-chart');
        if (!canvas) return;
        
        // Match buffer size to layout size to prevent stretching
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Fetch values for last 7 days
        const last7Days = [];
        const dailyCalSums = [];
        for (let i = -6; i <= 0; i++) {
            const dateObj = new Date();
            dateObj.setDate(dateObj.getDate() + i);
            last7Days.push(dateObj.toLocaleDateString('en-US', { weekday: 'short' }) + ' ' + dateObj.getDate());
            
            const dateKey = formatDateString(dateObj);
            const mealsKey = `munchin_meals_${currentUser.username}_${dateKey}`;
            const meals = JSON.parse(localStorage.getItem(mealsKey)) || [];
            const sum = meals.reduce((s, m) => s + m.calories, 0);
            dailyCalSums.push(sum);
        }

        // Draw graph borders/grids
        const paddingLeft = 35;
        const paddingRight = 15;
        const paddingTop = 20;
        const paddingBottom = 30;
        const chartWidth = canvas.width - paddingLeft - paddingRight;
        const chartHeight = canvas.height - paddingTop - paddingBottom;

        const maxVal = Math.max(3000, Math.max(...dailyCalSums) + 500);

        // Y Axis helper
        ctx.strokeStyle = '#EFEFF4';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#8E8E93';
        ctx.font = '10px Poppins';

        // Draw 3 horizontal helper grid lines
        for (let g = 0; g <= 3; g++) {
            const yVal = Math.round((maxVal / 3) * g);
            const y = canvas.height - paddingBottom - (chartHeight / 3) * g;
            
            ctx.beginPath();
            ctx.moveTo(paddingLeft, y);
            ctx.lineTo(canvas.width - paddingRight, y);
            ctx.stroke();

            ctx.fillText(yVal, 5, y + 3);
        }

        // Generate line path coordinates
        const coords = [];
        const xStep = chartWidth / (dailyCalSums.length - 1);
        
        dailyCalSums.forEach((sum, idx) => {
            const x = paddingLeft + idx * xStep;
            const y = canvas.height - paddingBottom - (sum / maxVal) * chartHeight;
            coords.push({ x, y, val: sum });
        });

        // Draw line fill gradient
        const fillGrad = ctx.createLinearGradient(0, paddingTop, 0, canvas.height - paddingBottom);
        fillGrad.addColorStop(0, 'rgba(240, 92, 59, 0.2)');
        fillGrad.addColorStop(1, 'rgba(240, 92, 59, 0.0)');

        ctx.beginPath();
        ctx.moveTo(coords[0].x, canvas.height - paddingBottom);
        coords.forEach(coord => ctx.lineTo(coord.x, coord.y));
        ctx.lineTo(coords[coords.length - 1].x, canvas.height - paddingBottom);
        ctx.closePath();
        ctx.fillStyle = fillGrad;
        ctx.fill();

        // Draw line path stroke
        ctx.strokeStyle = '#F05C3B';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(coords[0].x, coords[0].y);
        for (let i = 1; i < coords.length; i++) {
            ctx.lineTo(coords[i].x, coords[i].y);
        }
        ctx.stroke();

        // Draw points nodes
        coords.forEach(coord => {
            ctx.fillStyle = '#FFFFFF';
            ctx.strokeStyle = '#F05C3B';
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.arc(coord.x, coord.y, 4, 0, 2*Math.PI);
            ctx.fill();
            ctx.stroke();
        });

        // Draw X labels names
        ctx.fillStyle = '#8E8E93';
        coords.forEach((coord, idx) => {
            const name = last7Days[idx];
            ctx.textAlign = 'center';
            ctx.fillText(name, coord.x, canvas.height - 10);
        });

        // Compute average Kcal
        const avg = Math.round(dailyCalSums.reduce((s, v) => s + v, 0) / dailyCalSums.length);
        document.getElementById('reports-avg-val').textContent = avg;
    }

    // ----------------------------------
    // 9. Quick food selector setup
    // ----------------------------------
    let calendarCurrentDate = new Date();

    function renderCalendarGrid() {
        const daysGrid = document.getElementById('calendar-days-grid');
        const monthYearLabel = document.getElementById('calendar-month-year');
        if (!daysGrid || !monthYearLabel) return;

        daysGrid.innerHTML = '';
        
        const year = calendarCurrentDate.getFullYear();
        const month = calendarCurrentDate.getMonth();

        const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
        monthYearLabel.textContent = `${monthNames[month]} ${year}`;

        const firstDayIndex = new Date(year, month, 1).getDay();
        const totalDays = new Date(year, month + 1, 0).getDate();
        const prevTotalDays = new Date(year, month, 0).getDate();

        const todayKey = formatDateString(new Date());

        for (let i = firstDayIndex; i > 0; i--) {
            const dayNum = prevTotalDays - i + 1;
            const cell = document.createElement('div');
            cell.className = 'calendar-day-cell inactive';
            cell.textContent = dayNum;
            daysGrid.appendChild(cell);
        }

        for (let day = 1; day <= totalDays; day++) {
            const cellDate = new Date(year, month, day);
            const dateKey = formatDateString(cellDate);
            
            const cell = document.createElement('div');
            cell.className = 'calendar-day-cell';
            cell.textContent = day;

            if (dateKey === todayKey) {
                cell.classList.add('is-today');
            }
            if (dateKey === activeDate) {
                cell.classList.add('active');
            }

            cell.addEventListener('click', () => {
                activeDate = dateKey;
                setupHorizontalDates(cellDate);
                updateDashboardValues();
                document.getElementById('calendar-modal').classList.remove('active');
            });

            daysGrid.appendChild(cell);
        }

        const totalCellsSoFar = firstDayIndex + totalDays;
        const remainingCells = 42 - totalCellsSoFar;
        for (let i = 1; i <= remainingCells; i++) {
            const cell = document.createElement('div');
            cell.className = 'calendar-day-cell inactive';
            cell.textContent = i;
            daysGrid.appendChild(cell);
        }
    }

    function setupCalendarPicker() {
        const calendarToggleBtn = document.getElementById('calendar-toggle-btn');
        const calendarModal = document.getElementById('calendar-modal');
        const closeCalendarBtn = document.getElementById('close-calendar-modal');
        const prevMonthBtn = document.getElementById('calendar-prev-month');
        const nextMonthBtn = document.getElementById('calendar-next-month');

        if (calendarToggleBtn && calendarModal) {
            calendarToggleBtn.addEventListener('click', () => {
                if (activeDate) {
                    calendarCurrentDate = new Date(activeDate);
                } else {
                    calendarCurrentDate = new Date();
                }
                renderCalendarGrid();
                calendarModal.classList.add('active');
            });

            if (closeCalendarBtn) {
                closeCalendarBtn.addEventListener('click', () => {
                    calendarModal.classList.remove('active');
                });
            }

            calendarModal.addEventListener('click', (e) => {
                if (e.target === calendarModal) {
                    calendarModal.classList.remove('active');
                }
            });
        }

        if (prevMonthBtn) {
            prevMonthBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                calendarCurrentDate.setMonth(calendarCurrentDate.getMonth() - 1);
                renderCalendarGrid();
            });
        }

        if (nextMonthBtn) {
            nextMonthBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                calendarCurrentDate.setMonth(calendarCurrentDate.getMonth() + 1);
                renderCalendarGrid();
            });
        }
    }

    function setupExploreTabs() {
        const tabBtns = document.querySelectorAll('.explore-tab-btn');
        const views = document.querySelectorAll('.explore-view');
        
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const targetView = btn.dataset.view;
                views.forEach(view => {
                    view.classList.remove('active');
                    if (view.id === `explore-${targetView}-view`) {
                        view.classList.add('active');
                    }
                });

                if (targetView === 'new-dishes') {
                    loadExploreDishes();
                }
            });
        });

        const generateBtn = document.getElementById('planner-generate-btn');
        if (generateBtn) {
            generateBtn.addEventListener('click', generateMealRecipes);
        }

        const ingredientsInput = document.getElementById('planner-ingredients');
        if (ingredientsInput) {
            ingredientsInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    generateMealRecipes();
                }
            });
        }

        const aiToggle = document.getElementById('flavor-ai-toggle');
        if (aiToggle) {
            aiToggle.addEventListener('change', () => {
                loadExploreDishes();
            });
        }

        // Pill categories inside Custom Meal form
        const pills = document.querySelectorAll('.meal-category-select-pill .pill-btn');
        pills.forEach(pill => {
            pill.addEventListener('click', () => {
                const parent = pill.closest('.meal-category-select-pill');
                parent.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
                pill.classList.add('active');
            });
        });

        // Auto select all content on focus for manual entry inputs
        const customCalManual = document.getElementById('custom-food-calories');
        const customPManual = document.getElementById('custom-food-p');
        const customCManual = document.getElementById('custom-food-c');
        const customFManual = document.getElementById('custom-food-f');
        const customNameManual = document.getElementById('custom-food-name');

        [customCalManual, customPManual, customCManual, customFManual, customNameManual].forEach(input => {
            if (input) {
                input.addEventListener('focus', () => input.select());
            }
        });

        // Custom Manual Entry Submit
        const customSaveBtn = document.getElementById('custom-food-save-btn');
        if (customSaveBtn) {
            customSaveBtn.addEventListener('click', () => {
                const name = document.getElementById('custom-food-name').value.trim();
                const cal = parseInt(document.getElementById('custom-food-calories').value);
                const p = parseInt(document.getElementById('custom-food-p').value) || 0;
                const c = parseInt(document.getElementById('custom-food-c').value) || 0;
                const f = parseInt(document.getElementById('custom-food-f').value) || 0;

                if (!name) {
                    showToast('Enter a valid food name!', 'error');
                    return;
                }
                if (isNaN(cal) || cal <= 0) {
                    showToast('Calories must be positive!', 'error');
                    return;
                }

                const catPill = document.querySelector('#explore-manual-view .meal-category-select-pill .pill-btn.active');
                const targetCat = catPill ? catPill.dataset.cat : activeCategory;

                addMealItem(name, cal, p, c, f, targetCat);

                // Reset
                document.getElementById('custom-food-name').value = '';
                document.getElementById('custom-food-calories').value = '';
                document.getElementById('custom-food-p').value = '';
                document.getElementById('custom-food-c').value = '';
                document.getElementById('custom-food-f').value = '';
            });
        }
    }

    function loadExploreDishes() {
        const scrapedList = document.getElementById('scraped-dishes-list');
        scrapedList.innerHTML = `
            <div class="loading-state">
                <i class="fa-solid fa-spinner fa-spin text-coral"></i>
                <p>Scraping new culinary ideas...</p>
            </div>
        `;

        const aiToggle = document.getElementById('flavor-ai-toggle');
        const isAiEnabled = aiToggle && aiToggle.checked;

        if (isAiEnabled && currentUser) {
            // Collect last 7 days of meals
            const recentMeals = [];
            const oneDayMs = 24 * 60 * 60 * 1000;
            const today = new Date();
            
            for (let i = 0; i < 7; i++) {
                const date = new Date(today.getTime() - i * oneDayMs);
                const yyyy = date.getFullYear();
                const mm = String(date.getMonth() + 1).padStart(2, '0');
                const dd = String(date.getDate()).padStart(2, '0');
                const dateStr = `${yyyy}-${mm}-${dd}`;
                
                const mealsKey = `munchin_meals_${currentUser.username}_${dateStr}`;
                const loggedMeals = JSON.parse(localStorage.getItem(mealsKey) || '[]');
                loggedMeals.forEach(m => {
                    recentMeals.push({
                        name: m.name || '',
                        calories: parseInt(m.calories || 0),
                        protein: parseInt(m.protein || 0),
                        carbs: parseInt(m.carbs || 0),
                        fat: parseInt(m.fat || 0)
                    });
                });
            }

            // Collect user profile
            const configKey = `munchin_user_config_${currentUser.username}`;
            const userConfig = JSON.parse(localStorage.getItem(configKey) || '{}');
            
            const payload = {
                user_profile: {
                    gender: userConfig.gender || currentUser.gender || 'other',
                    age: parseInt(userConfig.age || currentUser.age || 25),
                    weight: parseFloat(userConfig.weight || currentUser.weight || 70),
                    goal: userConfig.goal || currentUser.goal || 'maintain',
                    target_calories: currentUser.targetCalories,
                    target_protein: currentUser.macros ? currentUser.macros.protein : 120,
                    target_carbs: currentUser.macros ? currentUser.macros.carbs : 200,
                    target_fat: currentUser.macros ? currentUser.macros.fats : 60
                },
                recent_meals: recentMeals
            };

            fetch('/api/v1/explore/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            })
            .then(res => {
                if (!res.ok) throw new Error("Flavor AI failed");
                return res.json();
            })
            .then(dishes => {
                renderExploreDishes(dishes, 'scraped-dishes-list');
                pushNotification(`Personalized Flavor AI match successfully calculated!`, 'recipe');
            })
            .catch(err => {
                console.error(err);
                scrapedList.innerHTML = `
                    <div class="loading-state">
                        <i class="fa-solid fa-circle-exclamation text-coral"></i>
                        <p>Flavor AI recommendation failed. Falling back...</p>
                        <button class="btn btn-secondary btn-block" id="retry-explore-btn" style="height: 34px; font-size:11px; margin-top:8px;">Try Again</button>
                    </div>
                `;
                const retryBtn = document.getElementById('retry-explore-btn');
                if (retryBtn) {
                    retryBtn.addEventListener('click', loadExploreDishes);
                }
            });
            return;
        }

        let url = '/api/v1/explore';
        if (currentUser) {
            const calories = currentUser.targetCalories || '';
            const protein = (currentUser.macros && currentUser.macros.protein) || '';
            const carbs = (currentUser.macros && currentUser.macros.carbs) || '';
            const fat = (currentUser.macros && currentUser.macros.fats) || '';
            url += `?calories=${calories}&protein=${protein}&carbs=${carbs}&fat=${fat}`;
        }

        fetch(url)
            .then(res => {
                if (!res.ok) throw new Error("Scraper failed");
                return res.json();
            })
            .then(dishes => {
                renderExploreDishes(dishes);
                pushNotification(`Discovered ${dishes.length} fresh healthy recipe ideas!`, 'recipe');
            })
            .catch(err => {
                console.error(err);
                scrapedList.innerHTML = `
                    <div class="loading-state">
                        <i class="fa-solid fa-circle-exclamation text-coral"></i>
                        <p>Failed to retrieve new dishes. Make sure the API server is running.</p>
                        <button class="btn btn-secondary btn-block" id="retry-explore-btn" style="height: 34px; font-size:11px; margin-top:8px;">Try Again</button>
                    </div>
                `;
                const retryBtn = document.getElementById('retry-explore-btn');
                if (retryBtn) {
                    retryBtn.addEventListener('click', loadExploreDishes);
                }
            });
    }

    function generateMealRecipes() {
        const ingredientsInput = document.getElementById('planner-ingredients');
        const ingredients = ingredientsInput ? ingredientsInput.value.trim() : '';
        if (!ingredients) {
            showToast('Please enter some ingredients first!', 'error');
            return;
        }

        const plannerList = document.getElementById('planner-dishes-list');
        plannerList.innerHTML = `
            <div class="loading-state" style="text-align: center; padding: 24px;">
                <i class="fa-solid fa-spinner fa-spin text-coral"></i>
                <p>Generating matching recipes from Ollama...</p>
            </div>
        `;

        let url = `/api/v1/explore/generate?ingredients=${encodeURIComponent(ingredients)}`;
        if (currentUser) {
            const calories = currentUser.targetCalories || '';
            const protein = (currentUser.macros && currentUser.macros.protein) || '';
            const carbs = (currentUser.macros && currentUser.macros.carbs) || '';
            const fat = (currentUser.macros && currentUser.macros.fats) || '';
            url += `&calories=${calories}&protein=${protein}&carbs=${carbs}&fat=${fat}`;
        }

        fetch(url)
            .then(res => {
                if (!res.ok) throw new Error("Planner failed");
                return res.json();
            })
            .then(dishes => {
                renderExploreDishes(dishes, 'planner-dishes-list');
                pushNotification(`Generated ${dishes.length} tailored recipe ideas!`, 'recipe');
            })
            .catch(err => {
                console.error(err);
                plannerList.innerHTML = `
                    <div class="loading-state" style="text-align: center; padding: 24px; color: var(--text-muted);">
                        <i class="fa-solid fa-circle-exclamation text-coral"></i>
                        <p>Failed to generate recipes. Please try again.</p>
                    </div>
                `;
            });
    }

    function renderExploreDishes(dishes, containerId = 'scraped-dishes-list') {
        const scrapedList = document.getElementById(containerId);
        if (!scrapedList) return;
        scrapedList.innerHTML = '';

        if (!dishes || dishes.length === 0) {
            scrapedList.innerHTML = `
                <div class="loading-state">
                    <p>No new dishes found.</p>
                </div>
            `;
            return;
        }

        dishes.forEach((dish, idx) => {
            const card = document.createElement('div');
            card.className = 'scraped-dish-card';
            
            // Check if match score or rationale exists (Flavor AI)
            const matchBadge = dish.match_score !== undefined ? `
                <div class="scraped-ai-badge" style="position: absolute; top: 12px; left: 12px; background: rgba(163, 217, 46, 0.95); color: #000; padding: 4px 8px; border-radius: 8px; font-size: 10px; font-weight: 700; display: flex; align-items: center; gap: 4px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); font-family: inherit; z-index: 2;">
                    <i class="fa-solid fa-brain"></i> ${dish.match_score}% Match
                </div>
            ` : '';

            const rationaleHtml = dish.rationale ? `
                <div class="scraped-ai-rationale" style="background: rgba(240, 92, 59, 0.05); border-left: 3px solid var(--accent-coral); padding: 8px 12px; border-radius: 8px; margin-bottom: 12px; font-size: 10.5px; color: var(--text-muted); line-height: 1.45; font-family: inherit; display: flex; align-items: flex-start; gap: 6px;">
                    <i class="fa-solid fa-sparkles" style="color: var(--accent-coral); margin-top: 2px;"></i> <span>${dish.rationale}</span>
                </div>
            ` : '';
            
            card.innerHTML = `
                <div class="scraped-dish-img-wrapper" style="position: relative;">
                    <img src="${dish.image_url}" alt="${dish.title}" onerror="this.src='https://images.unsplash.com/photo-1546069901-ba9599a7e63c?auto=format&fit=crop&q=80&w=400'">
                    ${matchBadge}
                </div>
                <div class="scraped-dish-info">
                    <div class="scraped-dish-title-row">
                        <h3>${dish.title}</h3>
                        <button class="scraped-readmore-btn" style="font-family: inherit;">Read Recipe <i class="fa-solid fa-chevron-down"></i></button>
                    </div>
                    <p class="scraped-dish-desc">${dish.description}</p>
                    ${rationaleHtml}
                    
                    <div class="recipe-expand-section" style="display: none; margin-top: 12px; border-top: 1px solid var(--border-color); padding-top: 12px; flex-direction: column; gap: 10px;">
                        <div class="recipe-ingredients-wrap">
                            <h4 style="font-size: 11px; font-weight: 700; color: var(--text-dark); margin-bottom: 4px;">Ingredients</h4>
                            <ul style="margin: 0; padding-left: 16px; font-size: 11px; color: var(--text-muted); line-height: 1.5; list-style-type: disc;">
                                ${(dish.recipe_ingredients || []).map(ing => `<li>${ing}</li>`).join('')}
                            </ul>
                        </div>
                        <div class="recipe-instructions-wrap">
                            <h4 style="font-size: 11px; font-weight: 700; color: var(--text-dark); margin-bottom: 4px;">Instructions</h4>
                            <ol style="margin: 0; padding-left: 16px; font-size: 11px; color: var(--text-muted); line-height: 1.5; list-style-type: decimal;">
                                ${(dish.recipe_instructions || []).map(step => `<li style="margin-bottom: 4px;">${step}</li>`).join('')}
                            </ol>
                        </div>
                        <a href="${dish.link}" target="_blank" style="font-size: 10px; color: var(--accent-coral); text-decoration: none; font-weight: 600; display: inline-flex; align-items: center; gap: 4px; margin-top: 4px; align-self: flex-start;">View original recipe <i class="fa-solid fa-arrow-up-right-from-square" style="font-size:8px;"></i></a>
                    </div>

                    <div class="scraped-dish-macros">
                        <div class="scraped-macro-item">
                            <span class="scraped-macro-lbl">Calories</span>
                            <span class="scraped-macro-val text-coral" style="color:#F05C3B">${dish.calories} kcal</span>
                        </div>
                        <div class="scraped-macro-item">
                            <span class="scraped-macro-lbl">Protein</span>
                            <span class="scraped-macro-val">${dish.protein}g</span>
                        </div>
                        <div class="scraped-macro-item">
                            <span class="scraped-macro-lbl">Carbs</span>
                            <span class="scraped-macro-val">${dish.carbs}g</span>
                        </div>
                        <div class="scraped-macro-item">
                            <span class="scraped-macro-lbl">Fats</span>
                            <span class="scraped-macro-val">${dish.fat}g</span>
                        </div>
                    </div>
                    <div class="scraped-dish-actions">
                        <div class="select-wrapper-light" style="position:relative;">
                            <select class="scraped-meal-cat-select form-select-light" style="width:100%; height:38px; border-radius:10px; border:1px solid #efeef4; padding: 0 10px; font-weight:600; font-size:12px; appearance:none;">
                                <option value="breakfast">Breakfast</option>
                                <option value="lunch" selected>Lunch</option>
                                <option value="dinner">Dinner</option>
                            </select>
                            <i class="fa-solid fa-chevron-down select-icon" style="position:absolute; right:12px; top:13px; font-size:10px; color:#8E8E93; pointer-events:none;"></i>
                        </div>
                        <button class="btn btn-green log-scraped-dish-btn"><i class="fa-solid fa-plus"></i> Log Meal</button>
                    </div>
                </div>
            `;

            // Toggle recipe section action
            const toggleBtn = card.querySelector('.scraped-readmore-btn');
            const expandSection = card.querySelector('.recipe-expand-section');
            if (toggleBtn && expandSection) {
                toggleBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    const isExpanded = expandSection.style.display === 'flex';
                    if (isExpanded) {
                        expandSection.style.display = 'none';
                        toggleBtn.innerHTML = `Read Recipe <i class="fa-solid fa-chevron-down"></i>`;
                    } else {
                        expandSection.style.display = 'flex';
                        toggleBtn.innerHTML = `Hide Recipe <i class="fa-solid fa-chevron-up"></i>`;
                    }
                });
            }

            // Log meal action
            card.querySelector('.log-scraped-dish-btn').addEventListener('click', () => {
                const categorySelect = card.querySelector('.scraped-meal-cat-select');
                const selectedCat = categorySelect.value;
                addMealItem(dish.title, dish.calories, dish.protein, dish.carbs, dish.fat, selectedCat);
            });

            // Convert category select to custom select dropdown
            convertSelectToCustom(card.querySelector('.scraped-meal-cat-select'));

            scrapedList.appendChild(card);
        });
    }

    // ----------------------------------
    // 10. AI Scanner Integrations
    // ----------------------------------
    const fileInputRevamp = document.getElementById('file-input-revamp');
    const dropzoneRevamp = document.getElementById('dropzone-revamp');
    const modeBtnsRevamp = document.querySelectorAll('.mode-btn-light');
    const modelSelectRevamp = document.getElementById('model-select-revamp');
    
    const progressPanelRevamp = document.getElementById('progress-panel-revamp');
    const progressStatusRevampText = document.getElementById('progress-status-revamp-text');
    const progressPercentRevamp = document.getElementById('progress-percent-revamp');
    const progressBarFillRevamp = document.getElementById('progress-bar-fill-revamp');
    const stepURevamp = document.getElementById('step-u-revamp');
    const stepCRevamp = document.getElementById('step-c-revamp');
    const stepNRevamp = document.getElementById('step-n-revamp');
    const stepSRevamp = document.getElementById('step-s-revamp');
    const stepPRevamp = document.getElementById('step-p-revamp');

    const resultPanelRevamp = document.getElementById('result-panel-revamp');
    const closeResultRevampBtn = document.getElementById('close-result-revamp-btn');
    const cancelResultRevampBtn = document.getElementById('res-cancel-btn');
    const saveResultRevampBtn = document.getElementById('res-save-btn');
    const resBackHomeBtn = document.getElementById('res-back-home-btn');
    
    const successModal = document.getElementById('success-modal');
    const successModalBtn = document.getElementById('success-modal-btn');
    const resultImgRevamp = document.getElementById('result-img-revamp');
    const overlayCanvasRevamp = document.getElementById('overlay-canvas-revamp');
    const overlayTagRevamp = document.getElementById('overlay-tag-revamp');
    const resultFoodNameRevamp = document.getElementById('result-food-name-revamp');
    const resultPortionRevamp = document.getElementById('result-portion-revamp');
    const resultConfVal = document.getElementById('result-conf-val');

    const resMacroCal = document.getElementById('res-macro-cal');
    const resMacroP = document.getElementById('res-macro-p');
    const resMacroC = document.getElementById('res-macro-c');
    const resMacroF = document.getElementById('res-macro-f');
    const resFillCal = document.getElementById('res-fill-cal');
    const resFillP = document.getElementById('res-fill-p');
    const resFillC = document.getElementById('res-fill-c');
    const resFillF = document.getElementById('res-fill-f');

    const resIngSection = document.getElementById('res-ing-section');
    const resIngList = document.getElementById('res-ing-list');
    const resDepthSection = document.getElementById('res-depth-section');
    const resDepthImg = document.getElementById('res-depth-img');

    let currentScanResult = null;
    let currentUploadedImageBase64Revamp = null;
    let activeScanPollInterval = null;
    let activeScanSocket = null;

    // Toggle scanning modes
    modeBtnsRevamp.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtnsRevamp.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Trigger file dialog
    dropzoneRevamp.addEventListener('click', () => {
        fileInputRevamp.click();
    });

    fileInputRevamp.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleScanFile(e.target.files[0]);
        }
    });

    function handleScanFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please upload an image file!', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (event) => {
            const dataUrl = event.target.result;
            currentUploadedImageBase64Revamp = dataUrl.split(',')[1];
            startAIScan(dataUrl);
        };
        reader.readAsDataURL(file);
    }

    function startAIScan(imageSrc) {
        const isAccurate = true;
        
        progressPanelRevamp.style.display = 'block';
        resultPanelRevamp.style.display = 'none';
        dropzoneRevamp.style.display = 'none';

        if (isAccurate) {
            stepSRevamp.style.display = 'flex';
            stepPRevamp.style.display = 'flex';
        } else {
            stepSRevamp.style.display = 'none';
            stepPRevamp.style.display = 'none';
        }

        resetScanSteps();

        runLiveAIScan(imageSrc, isAccurate);
    }

    function resetScanSteps() {
        const steps = [stepURevamp, stepCRevamp, stepNRevamp, stepSRevamp, stepPRevamp];
        steps.forEach(s => {
            s.className = 'step-item-light';
            s.querySelector('i').className = 'fa-regular fa-circle';
        });
        progressBarFillRevamp.style.width = '0%';
        progressPercentRevamp.textContent = '0%';
    }

    function updateScanStepUI(stepEl, status) {
        if (status === 'active') {
            stepEl.className = 'step-item-light active';
            stepEl.querySelector('i').className = 'fa-solid fa-circle-notch fa-spin';
        } else if (status === 'completed') {
            stepEl.className = 'step-item-light completed';
            stepEl.querySelector('i').className = 'fa-solid fa-circle-check';
        }
    }



    function runLiveAIScan(imageSrc, isAccurate) {
        isViewingSavedMeal = false;
        if (activeScanPollInterval) {
            clearInterval(activeScanPollInterval);
            activeScanPollInterval = null;
        }
        if (activeScanSocket) {
            activeScanSocket.close();
            activeScanSocket = null;
        }

        updateScanStepUI(stepURevamp, 'active');
        progressStatusRevampText.textContent = 'Uploading picture to gateway...';
        progressBarFillRevamp.style.width = '10%';
        progressPercentRevamp.textContent = '10%';

        const selectedModel = modelSelectRevamp.value;
        const payload = {
            image_base64: currentUploadedImageBase64Revamp,
            mode: isAccurate ? 'accurate' : 'fast',
            models: selectedModel ? [selectedModel] : null
        };

        fetch('/api/v1/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => {
            if (!res.ok) throw new Error('API failed');
            return res.json();
        })
        .then(job => {
            updateScanStepUI(stepURevamp, 'completed');
            updateScanStepUI(stepCRevamp, 'active');
            progressStatusRevampText.textContent = 'Running classification layers...';
            progressBarFillRevamp.style.width = '25%';
            progressPercentRevamp.textContent = '25%';

            connectScanJobWS(job.job_id, isAccurate, imageSrc);
        })
        .catch(() => {
            showToast('Live backend analysis failed! Check API connection.', 'error');
            resetScanDropzone();
        });
    }

    function connectScanJobWS(jobId, isAccurate, imageSrc) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v1/jobs/${jobId}/stream`;

        const socket = new WebSocket(wsUrl);
        activeScanSocket = socket;

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.error) {
                showToast(`Analysis failed: ${data.error}`, 'error');
                showErrorModal(data.error);
                resetScanDropzone();
                socket.close();
                return;
            }

            // Parse progress percentage
            let prog = 25;
            if (data.progress && typeof data.progress === 'object') {
                const p = data.progress;
                if (p.classification === 'completed') prog = 30;
                if (p.nutrition === 'completed') prog = 60;
                if (isAccurate) {
                    if (p.detection === 'completed') prog = 85;
                    if (p.portion === 'completed') prog = 95;
                }
            } else if (typeof data.progress === 'number') {
                prog = data.progress;
            }
            if (data.status === 'completed') {
                prog = 100;
            }

            progressBarFillRevamp.style.width = `${prog}%`;
            progressPercentRevamp.textContent = `${prog}%`;

            if (prog >= 30) {
                updateScanStepUI(stepCRevamp, 'completed');
                updateScanStepUI(stepNRevamp, 'active');
                progressStatusRevampText.textContent = 'Matching nutrient records...';
            }
            if (isAccurate && prog >= 60) {
                updateScanStepUI(stepNRevamp, 'completed');
                updateScanStepUI(stepSRevamp, 'active');
                progressStatusRevampText.textContent = 'Segmenting food boundaries...';
            }
            if (isAccurate && prog >= 85) {
                updateScanStepUI(stepSRevamp, 'completed');
                updateScanStepUI(stepPRevamp, 'active');
                progressStatusRevampText.textContent = 'Estimating portion thickness...';
            }

            if (data.status === 'completed') {
                socket.close();
                if (isAccurate) updateScanStepUI(stepPRevamp, 'completed');
                else updateScanStepUI(stepNRevamp, 'completed');

                const mapped = translateJobResult(data.result, imageSrc);
                setTimeout(() => {
                    displayScanResult(mapped, isAccurate);
                }, 500);
            } else if (data.status === 'failed') {
                socket.close();
                showToast(`Analysis failed: ${data.error}`, 'error');
                showErrorModal(data.error);
                resetScanDropzone();
            }
        };

        socket.onerror = (error) => {
            console.error('WebSocket connection error. Falling back to HTTP polling...', error);
            socket.close();
            pollScanJob(jobId, isAccurate, imageSrc);
        };

        socket.onclose = () => {
            if (activeScanSocket === socket) {
                activeScanSocket = null;
            }
        };
    }

    function pollScanJob(jobId, isAccurate, imageSrc) {
        let count = 0;
        const maxPolls = 60;
        activeScanPollInterval = setInterval(() => {
            count++;
            if (count > maxPolls) {
                clearInterval(activeScanPollInterval);
                showToast('Job polling timed out!', 'error');
                resetScanDropzone();
                return;
            }

            fetch(`/api/v1/jobs/${jobId}`)
                .then(res => {
                    if (!res.ok) throw new Error();
                    return res.json();
                })
                .then(job => {
                    // Parse progress percentage from backend step completion indicators
                    let prog = 25;
                    if (job.progress && typeof job.progress === 'object') {
                        const p = job.progress;
                        if (p.classification === 'completed') prog = 30;
                        if (p.nutrition === 'completed') prog = 60;
                        if (isAccurate) {
                            if (p.detection === 'completed') prog = 85;
                            if (p.portion === 'completed') prog = 95;
                        }
                    } else if (typeof job.progress === 'number') {
                        prog = job.progress;
                    }
                    if (job.status === 'completed') {
                        prog = 100;
                    }

                    progressBarFillRevamp.style.width = `${prog}%`;
                    progressPercentRevamp.textContent = `${prog}%`;

                    if (prog >= 30) {
                        updateScanStepUI(stepCRevamp, 'completed');
                        updateScanStepUI(stepNRevamp, 'active');
                        progressStatusRevampText.textContent = 'Matching nutrient records...';
                    }
                    if (isAccurate && prog >= 60) {
                        updateScanStepUI(stepNRevamp, 'completed');
                        updateScanStepUI(stepSRevamp, 'active');
                        progressStatusRevampText.textContent = 'Segmenting food boundaries...';
                    }
                    if (isAccurate && prog >= 85) {
                        updateScanStepUI(stepSRevamp, 'completed');
                        updateScanStepUI(stepPRevamp, 'active');
                        progressStatusRevampText.textContent = 'Estimating portion thickness...';
                    }

                    if (job.status === 'completed') {
                        clearInterval(activeScanPollInterval);
                        if (isAccurate) updateScanStepUI(stepPRevamp, 'completed');
                        else updateScanStepUI(stepNRevamp, 'completed');

                        // Map results
                        const mapped = translateJobResult(job.result, imageSrc);
                        setTimeout(() => {
                            displayScanResult(mapped, isAccurate);
                        }, 500);
                    } else if (job.status === 'failed') {
                        clearInterval(activeScanPollInterval);
                        showToast(`Analysis failed: ${job.error}`, 'error');
                        showErrorModal(job.error);
                        resetScanDropzone();
                    }
                })
                .catch(() => {});
        }, 1500);
    }

    function translateJobResult(apiResult, imageSrc) {
        let cal = 400;
        let protein = 15;
        let carbs = 45;
        let fat = 10;
        
        let multiplier = 1.0;
        if (apiResult.portion && typeof apiResult.portion === 'object') {
            const estWeight = apiResult.portion.estimated_weight_grams;
            if (estWeight !== undefined && estWeight !== null) {
                multiplier = estWeight / 100.0;
            }
        }
        
        if (apiResult.nutrition) {
            const getVal = (field, alternateField, fallback) => {
                const item = apiResult.nutrition[field] !== undefined ? apiResult.nutrition[field] : apiResult.nutrition[alternateField];
                if (item === undefined || item === null) return fallback;
                if (typeof item === 'object' && item.value !== undefined) return item.value;
                if (typeof item === 'number') return item;
                return fallback;
            };
            cal = Math.round(getVal('calories', 'energy', 400) * multiplier);
            protein = Math.round(getVal('protein', 'protein', 15) * multiplier);
            carbs = Math.round(getVal('carbohydrates', 'carbohydrate', 45) * multiplier);
            fat = Math.round(getVal('total_fat', 'fat', 10) * multiplier);
        }
        
        const ingredients = (apiResult.ingredients || []).map(ing => {
            const weight = ing.weight_g || Math.round((ing.mask_area_ratio || ing.area_ratio || 0) * 200) || 50;
            let ingCal = ing.calories || 0;
            if (ing.nutrition) {
                const getVal = (field, alternateField, fallback) => {
                    const item = ing.nutrition[field] !== undefined ? ing.nutrition[field] : ing.nutrition[alternateField];
                    if (item === undefined || item === null) return fallback;
                    if (typeof item === 'object' && item.value !== undefined) return item.value;
                    if (typeof item === 'number') return item;
                    return fallback;
                };
                const calPer100g = getVal('calories', 'energy', 0);
                ingCal = (calPer100g * weight) / 100;
            }
            return {
                name: ing.class_name || ing.name || ing.label,
                weight: weight,
                calories: Math.round(ingCal)
            };
        });

        return {
            name: apiResult.class_name,
            confidence: Math.round(apiResult.confidence * 100),
            portion: apiResult.portion ? (typeof apiResult.portion === 'object' ? `${((apiResult.portion.estimated_weight_grams || 300) / (apiResult.portion.typical_portion_grams || 300)).toFixed(1)} portions` : `${Number(apiResult.portion).toFixed(1)} portions`) : '1.0 portion',
            calories: cal,
            protein: protein,
            carbs: carbs,
            fat: fat,
            ingredients: ingredients,
            overlay_url: apiResult.overlay_url,
            depth_url: apiResult.depth_map_url,
            image_src: imageSrc
        };
    }

    function displayScanResult(result, isAccurate) {
        currentScanResult = result;

        progressPanelRevamp.style.display = 'none';
        resultPanelRevamp.style.display = 'block';

        resultImgRevamp.src = result.image_src;
        resultFoodNameRevamp.textContent = result.name;
        resultPortionRevamp.innerHTML = `<i class="fa-solid fa-calculator text-coral"></i> Estimated portion: <strong>${result.portion}</strong>`;
        resultConfVal.textContent = `${result.confidence}%`;

        // Bounding canvas overlays
        drawOverlayCanvas(result.name, isAccurate);

        // Populate macros
        resMacroCal.textContent = `${result.calories} kcal`;
        resMacroP.textContent = `${result.protein}g`;
        resMacroC.textContent = `${result.carbs}g`;
        resMacroF.textContent = `${result.fat}g`;

        resFillCal.style.width = `${Math.min(100, (result.calories / 800) * 100)}%`;
        resFillP.style.width = `${Math.min(100, (result.protein / 40) * 100)}%`;
        resFillC.style.width = `${Math.min(100, (result.carbs / 100) * 100)}%`;
        resFillF.style.width = `${Math.min(100, (result.fat / 30) * 100)}%`;

        // Show/hide saved vs active scanner footer buttons
        if (isViewingSavedMeal) {
            cancelResultRevampBtn.style.display = 'none';
            saveResultRevampBtn.style.display = 'none';
            if (resBackHomeBtn) resBackHomeBtn.style.display = 'block';
            
            const catSelectGroup = document.getElementById('result-meal-cat-select')?.closest('.form-group');
            if (catSelectGroup) catSelectGroup.style.display = 'none';
        } else {
            cancelResultRevampBtn.style.display = 'block';
            saveResultRevampBtn.style.display = 'block';
            if (resBackHomeBtn) resBackHomeBtn.style.display = 'none';
            
            const catSelectGroup = document.getElementById('result-meal-cat-select')?.closest('.form-group');
            if (catSelectGroup) catSelectGroup.style.display = 'block';
        }

        // Ingredients details list
        if (isAccurate && result.ingredients && result.ingredients.length > 0) {
            resIngSection.style.display = 'block';
            resIngList.innerHTML = '';
            
            const colors = ['#F05C3B', '#F5C542', '#5BC585', '#B6A6E8', '#00f2fe'];
            result.ingredients.forEach((ing, idx) => {
                const color = colors[idx % colors.length];
                const item = document.createElement('div');
                item.className = 'ingredient-item';
                // Inline styling for simplicity in light theme item matching
                item.style.backgroundColor = '#f8f9fa';
                item.style.border = '1px solid #efeef4';
                item.style.borderRadius = '12px';
                item.style.padding = '8px 12px';
                item.style.fontSize = '11px';
                item.style.display = 'flex';
                item.style.justifyContent = 'space-between';
                item.style.marginBottom = '6px';

                item.innerHTML = `
                    <div style="display:flex; align-items:center; gap:6px;">
                        <span style="width:6px; height:6px; border-radius:50%; background-color:${color};"></span>
                        <strong style="color:#2E2E2E;">${ing.name}</strong>
                        <span style="color:#8E8E93;">(${ing.weight}g)</span>
                    </div>
                    <span style="font-weight:600; color:#8E8E93;">${ing.calories} kcal</span>
                `;
                resIngList.appendChild(item);
            });
        } else {
            resIngSection.style.display = 'none';
        }

        // Depth section
        if (isAccurate && (result.depth_url || result.isMock)) {
            resDepthSection.style.display = 'block';
            overlayTagRevamp.style.display = 'block';
            if (result.isMock) {
                resDepthImg.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="225" viewBox="0 0 300 225"><defs><radialGradient id="g" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="%23ff00ff"/><stop offset="50%" stop-color="%230000ff"/><stop offset="100%" stop-color="%23000000"/></radialGradient></defs><rect width="300" height="225" fill="url(%23g)"/></svg>';
            } else {
                resDepthImg.src = result.depth_url;
            }
        } else {
            resDepthSection.style.display = 'none';
            overlayTagRevamp.style.display = 'none';
        }
    }

    function drawOverlayCanvas(foodName, isAccurate) {
        const ctx = overlayCanvasRevamp.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvasRevamp.width, overlayCanvasRevamp.height);

        if (!isAccurate) return;

        overlayCanvasRevamp.width = overlayCanvasRevamp.offsetWidth;
        overlayCanvasRevamp.height = overlayCanvasRevamp.offsetHeight;

        const w = overlayCanvasRevamp.width;
        const h = overlayCanvasRevamp.height;

        ctx.lineWidth = 2.5;

        if (foodName.includes('Phở') || foodName.includes('Pho')) {
            // Broth segment overlay
            ctx.strokeStyle = '#B6A6E8';
            ctx.fillStyle = 'rgba(182, 166, 232, 0.15)';
            ctx.beginPath();
            ctx.arc(w/2, h/2 + 5, w/3.2, 0, 2*Math.PI);
            ctx.fill();
            ctx.stroke();

            // Beef segment overlay
            ctx.strokeStyle = '#F05C3B';
            ctx.fillStyle = 'rgba(240, 92, 59, 0.2)';
            ctx.beginPath();
            ctx.arc(w/2 + 25, h/2 - 15, 20, 0, 2*Math.PI);
            ctx.arc(w/2 + 5, h/2 - 20, 18, 0, 2*Math.PI);
            ctx.fill();
            ctx.stroke();
        } else {
            // Baguette crust
            ctx.strokeStyle = '#F5C542';
            ctx.fillStyle = 'rgba(245, 197, 66, 0.15)';
            ctx.beginPath();
            ctx.ellipse(w/2, h/2, w/2.9, h/5.2, -Math.PI/12, 0, 2*Math.PI);
            ctx.fill();
            ctx.stroke();
        }
    }

    function resetScanDropzone() {
        if (activeScanPollInterval) {
            clearInterval(activeScanPollInterval);
            activeScanPollInterval = null;
        }
        dropzoneRevamp.style.display = 'block';
        progressPanelRevamp.style.display = 'none';
        resultPanelRevamp.style.display = 'none';
        fileInputRevamp.value = '';
        currentUploadedImageBase64Revamp = null;
        currentScanResult = null;

        if (isViewingSavedMeal) {
            isViewingSavedMeal = false;
            showAppScreen('home');
        }
    }

    closeResultRevampBtn.addEventListener('click', resetScanDropzone);
    cancelResultRevampBtn.addEventListener('click', resetScanDropzone);
    if (resBackHomeBtn) {
        resBackHomeBtn.addEventListener('click', resetScanDropzone);
    }

    if (successModalBtn) {
        successModalBtn.addEventListener('click', () => {
            if (successModal) successModal.classList.remove('active');
            resetScanDropzone();
            showAppScreen('home'); // Go back home
        });
    }
    if (successModal) {
        successModal.addEventListener('click', (e) => {
            if (e.target === successModal) {
                successModal.classList.remove('active');
                resetScanDropzone();
                showAppScreen('home');
            }
        });
    }

    const cancelProgressRevampBtn = document.getElementById('cancel-progress-revamp-btn');
    if (cancelProgressRevampBtn) {
        cancelProgressRevampBtn.addEventListener('click', resetScanDropzone);
    }

    // Save scan to database log
    saveResultRevampBtn.addEventListener('click', () => {
        if (currentScanResult) {
            // Find active pill category inside Scan Results
            const catPill = document.querySelector('#result-meal-cat-select .pill-btn.active');
            const targetCat = catPill ? catPill.dataset.cat : activeCategory;

            addMealItem(
                currentScanResult.name,
                currentScanResult.calories,
                currentScanResult.protein,
                currentScanResult.carbs,
                currentScanResult.fat,
                targetCat,
                currentScanResult.image_src,
                currentScanResult.portion,
                currentScanResult.ingredients,
                currentScanResult.depth_url
            );
        }
    });

    // Check backend health for revamp view
    function checkBackendHealth() {
        const badge = document.getElementById('backend-status-revamp');
        badge.className = 'badge';
        badge.innerHTML = `<span class="status-dot"></span>Checking...`;

        fetch('/api/v1/health')
            .then(res => {
                if (!res.ok) throw new Error();
                return res.json();
            })
            .then(data => {
                if (data.status === 'ok' || data.status === 'degraded') {
                    databaseStatus = 'online';
                    badge.className = 'badge online';
                    badge.innerHTML = `<span class="status-dot"></span>Online`;
                } else throw new Error();
            })
            .catch(() => {
                databaseStatus = 'offline';
                badge.className = 'badge offline';
                badge.innerHTML = `<span class="status-dot"></span>Offline`;
            });
    }

    function fetchAvailableModels() {
        convertSelectToCustom(modelSelectRevamp);

        fetch('/api/v1/models')
            .then(res => {
                if (!res.ok) throw new Error();
                return res.json();
            })
            .then(models => {
                modelSelectRevamp.innerHTML = '<option value="">Default Model</option>';
                models.forEach(model => {
                    const opt = document.createElement('option');
                    opt.value = model.name;
                    opt.textContent = model.name.startsWith('ollama:')
                        ? `Ollama: ${model.name.split(':')[1].toUpperCase()}`
                        : model.name.startsWith('gemini:')
                            ? `Gemini: ${model.name.split(':')[1].toUpperCase()}`
                            : model.name.split('/').pop().replace('.pth', '').replace('.onnx', '');
                    modelSelectRevamp.appendChild(opt);
                });
                convertSelectToCustom(modelSelectRevamp);
            })
            .catch(() => {
                convertSelectToCustom(modelSelectRevamp);
            });
    }

    // Result definitions


    // ----------------------------------
    // 11. Custom Toast Utility
    // ----------------------------------
    function showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icon = type === 'success' ? 
            '<i class="fa-solid fa-circle-check" style="color:#A3D92E"></i>' : 
            '<i class="fa-solid fa-circle-exclamation" style="color:#F05C3B"></i>';

        toast.innerHTML = `${icon} <span>${message}</span>`;
        const container = document.getElementById('toast-container');
        if (container) {
            container.appendChild(toast);
        }

        setTimeout(() => {
            toast.style.animation = 'fadeIn 0.3s ease-out reverse';
            setTimeout(() => {
                toast.remove();
            }, 300);
        }, 3000);
    }

    // Deletion Modal Button Handlers
    document.getElementById('confirm-delete-meal-btn').addEventListener('click', () => {
        const id = document.getElementById('delete-meal-id').value;
        removeMeal(id);
        document.getElementById('delete-meal-modal').classList.remove('active');
    });

    document.getElementById('confirm-delete-workout-btn').addEventListener('click', () => {
        const id = document.getElementById('delete-workout-id').value;
        removeWorkout(id);
        document.getElementById('delete-workout-modal').classList.remove('active');
    });

    document.getElementById('confirm-delete-weight-btn').addEventListener('click', () => {
        const date = document.getElementById('delete-weight-date').value;
        removeWeightLog(date);
        document.getElementById('delete-weight-modal').classList.remove('active');
    });

    const closeDeleteModalPairs = [
        { btn: 'close-delete-meal-btn', modal: 'delete-meal-modal' },
        { btn: 'cancel-delete-meal-btn', modal: 'delete-meal-modal' },
        { btn: 'close-delete-workout-btn', modal: 'delete-workout-modal' },
        { btn: 'cancel-delete-workout-btn', modal: 'delete-workout-modal' },
        { btn: 'close-delete-weight-btn', modal: 'delete-weight-modal' },
        { btn: 'cancel-delete-weight-btn', modal: 'delete-weight-modal' }
    ];
    closeDeleteModalPairs.forEach(pair => {
        document.getElementById(pair.btn).addEventListener('click', () => {
            document.getElementById(pair.modal).classList.remove('active');
        });
    });

    // Trigger initialization
    init();
});
