// Admin Dashboard Controller - Vanilla JS
document.addEventListener("DOMContentLoaded", () => {
  // --- STATE ---
  let token = localStorage.getItem("admin_token") || null;
  let username = localStorage.getItem("admin_username") || null;
  let role = localStorage.getItem("admin_role") || null;
  let jobChart = null;

  // --- DOM ELEMENTS ---
  const authView = document.getElementById("auth-view");
  const dashboardView = document.getElementById("dashboard-view");
  const loginForm = document.getElementById("login-form");
  const logoutBtn = document.getElementById("btn-logout");
  const displayUsername = document.getElementById("display-username");
  const displayRole = document.getElementById("display-role");

  // Metrics
  const statTotalUsers = document.getElementById("stat-total-users");
  const statActiveSessions = document.getElementById("stat-active-sessions");
  const statTotalJobs = document.getElementById("stat-total-jobs");
  const statFailedJobs = document.getElementById("stat-failed-jobs");
  const dbHealthBadge = document.getElementById("db-health-badge");

  // User Table
  const usersTableBody = document.getElementById("users-table-body");
  const btnOpenCreateModal = document.getElementById("btn-open-create-modal");

  // Modals & Forms
  const createModal = document.getElementById("create-user-modal");
  const createForm = document.getElementById("create-user-form");
  const editModal = document.getElementById("edit-user-modal");
  const editForm = document.getElementById("edit-user-form");
  const deleteModal = document.getElementById("delete-confirm-modal");
  const btnConfirmDelete = document.getElementById("btn-confirm-delete");

  // Toast
  const toastContainer = document.getElementById("toast-container");

  // --- TOAST NOTIFICATIONS ---
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    let icon = "fa-info-circle";
    if (type === "success") icon = "fa-check-circle";
    if (type === "error") icon = "fa-exclamation-circle";
    
    toast.innerHTML = `
      <i class="fa-solid ${icon}"></i>
      <span>${message}</span>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto remove
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(-10px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // --- API HELPER FUNCTIONS ---
  async function apiRequest(endpoint, method = "GET", body = null) {
    const headers = {
      "Content-Type": "application/json"
    };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    
    const config = {
      method,
      headers
    };
    if (body) {
      config.body = JSON.stringify(body);
    }

    try {
      const response = await fetch(`/api/v1${endpoint}`, config);
      if (response.status === 401) {
        if (endpoint === "/auth/login") {
          const data = await response.json();
          showToast(data.detail || "Sai tên đăng nhập hoặc mật khẩu!", "error");
          return null;
        }
        // Expired or invalid session
        handleLogoutLocal();
        showToast("Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.", "error");
        return null;
      }
      
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Đã xảy ra lỗi không xác định");
      }
      return data;
    } catch (error) {
      showToast(error.message, "error");
      return null;
    }
  }

  // --- SCREEN SWITCHING ---
  function checkAuth() {
    if (token && role === "admin") {
      authView.style.display = "none";
      dashboardView.style.display = "block";
      displayUsername.textContent = username;
      displayRole.textContent = role.toUpperCase();
      loadDashboardData();
    } else {
      authView.style.display = "flex";
      dashboardView.style.display = "none";
    }
  }

  function handleLogoutLocal() {
    token = null;
    username = null;
    role = null;
    localStorage.removeItem("admin_token");
    localStorage.removeItem("admin_username");
    localStorage.removeItem("admin_role");
    checkAuth();
  }

  // --- LOGIN & LOGOUT ---
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const u = document.getElementById("login-username").value.trim().toLowerCase();
    const p = document.getElementById("login-password").value;

    const data = await apiRequest("/auth/login", "POST", { username: u, password: p });
    if (data) {
      if (data.role !== "admin") {
        showToast("Quyền truy cập bị từ chối. Chỉ dành cho Quản trị viên.", "error");
        return;
      }
      token = data.session_token;
      username = data.username;
      role = data.role;
      localStorage.setItem("admin_token", token);
      localStorage.setItem("admin_username", username);
      localStorage.setItem("admin_role", role);
      showToast("Đăng nhập thành công!", "success");
      checkAuth();
    }
  });

  logoutBtn.addEventListener("click", async () => {
    await apiRequest("/auth/logout", "POST");
    handleLogoutLocal();
    showToast("Đã đăng xuất.", "info");
  });

  // --- LOAD DASHBOARD DATA ---
  async function loadDashboardData() {
    const stats = await apiRequest("/admin/stats");
    if (stats) {
      statTotalUsers.textContent = stats.total_users;
      statActiveSessions.textContent = stats.active_sessions;
      statTotalJobs.textContent = stats.total_jobs;
      statFailedJobs.textContent = stats.failed_jobs;
      dbHealthBadge.className = `badge badge-${stats.db_status === 'healthy' ? 'active' : 'blocked'}`;
      dbHealthBadge.textContent = stats.db_status === 'healthy' ? 'Hoạt động' : 'Degraded';

      // Redraw chart
      drawChart(stats.completed_jobs, stats.failed_jobs, stats.total_jobs - stats.completed_jobs - stats.failed_jobs);
    }

    const users = await apiRequest("/admin/users");
    if (users) {
      populateUsersTable(users);
    }
  }

  // --- CHART RENDERING ---
  function drawChart(completed, failed, pending) {
    const ctx = document.getElementById("jobChart").getContext("2d");
    
    if (jobChart) {
      jobChart.destroy();
    }

    // If zero total jobs, mock a placeholder
    const isZero = (completed + failed + pending) === 0;
    const dataValues = isZero ? [1] : [completed, failed, pending];
    const dataLabels = isZero ? ["Chưa có dữ liệu"] : ["Thành công", "Thất bại", "Đang xử lý"];
    const backgroundColors = isZero ? ["#1f2937"] : ["#10b981", "#ef4444", "#f59e0b"];

    jobChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: dataLabels,
        datasets: [{
          data: dataValues,
          backgroundColor: backgroundColors,
          borderWidth: 0,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: true,
            position: "bottom",
            labels: {
              color: "#f3f4f6",
              boxWidth: 12,
              padding: 15,
              font: {
                family: "'Outfit', sans-serif"
              }
            }
          }
        },
        cutout: "70%"
      }
    });
  }

  // --- POPULATE USERS TABLE ---
  function populateUsersTable(users) {
    usersTableBody.innerHTML = "";
    if (users.length === 0) {
      usersTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Không tìm thấy người dùng nào.</td></tr>`;
      return;
    }

    users.forEach(user => {
      const tr = document.createElement("tr");
      
      const roleBadge = user.role === "admin" ? "badge-admin" : "badge-user";
      const statusBadge = user.is_active ? "badge-active" : "badge-blocked";
      const statusText = user.is_active ? "Active" : "Blocked";
      
      const createdDate = new Date(user.created_at).toLocaleDateString("vi-VN", {
        year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"
      });

      // Disable block/delete actions for self or main 'admin'
      const isSelf = user.username === username || user.username === "admin";
      const actionDisabled = isSelf ? "disabled style='opacity:0.3; cursor:not-allowed;'" : "";

      tr.innerHTML = `
        <td style="font-weight: 500;">${user.username}</td>
        <td><span class="badge ${roleBadge}">${user.role.toUpperCase()}</span></td>
        <td><span class="badge ${statusBadge}">${statusText}</span></td>
        <td style="color: var(--text-secondary);">${createdDate}</td>
        <td>
          <div class="action-buttons">
            <button class="btn-icon edit" onclick="openEditModal('${user.id}', '${user.username}', '${user.role}', ${user.is_active})">
              <i class="fa-solid fa-pen-to-square"></i>
            </button>
            <button class="btn-icon block" ${actionDisabled} onclick="toggleUserStatus('${user.id}', '${user.username}')">
              <i class="fa-solid ${user.is_active ? 'fa-user-slash' : 'fa-user-check'}"></i>
            </button>
            <button class="btn-icon delete" ${actionDisabled} onclick="openDeleteConfirmModal('${user.id}', '${user.username}')">
              <i class="fa-solid fa-trash-can"></i>
            </button>
          </div>
        </td>
      `;
      usersTableBody.appendChild(tr);
    });
  }

  // --- MODAL TRIGGERS & CLOSES ---
  btnOpenCreateModal.addEventListener("click", () => {
    createForm.reset();
    createModal.classList.add("active");
  });

  document.querySelectorAll(".modal-close-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".modal-overlay").forEach(modal => modal.classList.remove("active"));
    });
  });

  // Expose triggers to window so inline onclick handlers in table can access them
  window.openEditModal = (id, username, role, active) => {
    document.getElementById("edit-user-id").value = id;
    document.getElementById("edit-username").value = username;
    document.getElementById("edit-password").value = "";
    document.getElementById("edit-role").value = role;
    document.getElementById("edit-active").checked = active;
    editModal.classList.add("active");
  };

  window.openDeleteConfirmModal = (id, username) => {
    document.getElementById("delete-user-id").value = id;
    document.getElementById("delete-username-label").textContent = username;
    deleteModal.classList.add("active");
  };

  // --- FORM SUBMISSIONS ---
  
  // Create User
  createForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const u = document.getElementById("create-username").value.trim().toLowerCase();
    const p = document.getElementById("create-password").value;
    const r = document.getElementById("create-role").value;

    const data = await apiRequest("/admin/users", "POST", { username: u, password: p, role: r });
    if (data) {
      showToast(`Đã tạo thành công người dùng "${u}".`, "success");
      createModal.classList.remove("active");
      loadDashboardData();
    }
  });

  // Edit User
  editForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const id = document.getElementById("edit-user-id").value;
    const u = document.getElementById("edit-username").value.trim().toLowerCase();
    const p = document.getElementById("edit-password").value;
    const r = document.getElementById("edit-role").value;
    const active = document.getElementById("edit-active").checked;

    const payload = { username: u, role: r, is_active: active };
    if (p !== "") {
      payload.password = p;
    }

    const data = await apiRequest(`/admin/users/${id}`, "PUT", payload);
    if (data) {
      showToast(`Đã cập nhật thông tin người dùng "${u}".`, "success");
      editModal.classList.remove("active");
      loadDashboardData();
    }
  });

  // Toggle User Active Status (Block / Unblock)
  window.toggleUserStatus = async (id, username) => {
    const data = await apiRequest(`/admin/users/${id}/toggle-status`, "POST");
    if (data) {
      const statusText = data.is_active ? "mở khóa" : "khóa";
      showToast(`Đã ${statusText} tài khoản "${username}".`, "info");
      loadDashboardData();
    }
  };

  // Delete User
  btnConfirmDelete.addEventListener("click", async () => {
    const id = document.getElementById("delete-user-id").value;
    const username = document.getElementById("delete-username-label").textContent;

    const data = await apiRequest(`/admin/users/${id}`, "DELETE");
    if (data) {
      showToast(`Đã xóa tài khoản "${username}".`, "success");
      deleteModal.classList.remove("active");
      loadDashboardData();
    }
  });

  // --- INITIAL CHECK ---
  checkAuth();
});
